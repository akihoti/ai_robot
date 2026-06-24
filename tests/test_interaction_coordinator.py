from __future__ import annotations

import asyncio
import logging
import unittest

from ai_robot_edge.config import VisionConfig
from ai_robot_edge.events import ActionName, VisionEvent, VisionEventType
from ai_robot_edge.interaction.coordinator import InteractionCoordinator
from ai_robot_edge.session import EdgeSessionState, SessionController


class InteractionCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 10.0

    async def test_absent_events_queue_only_one_idle_action_until_person_returns(self) -> None:
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        action_queue = asyncio.Queue()
        conversation_queue = asyncio.Queue()
        coordinator = InteractionCoordinator(
            vision_config=VisionConfig(
                detector="simulated",
                person_threshold=0.5,
                stable_frames=1,
                welcome_cooldown_seconds=30,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=SessionController(),
            disarm_listening=lambda: None,
            idle_return_to_center_seconds=0,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            for _ in range(5):
                await vision_queue.put(
                    VisionEvent(event_type=VisionEventType.PERSON_ABSENT, confidence=0)
                )
                await asyncio.sleep(0.01)

            self.assertEqual(action_queue.qsize(), 1)
            intent = action_queue.get_nowait()
            self.assertEqual(intent.name, ActionName.IDLE)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_absent_events_do_not_queue_idle_while_session_is_active(self) -> None:
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        action_queue = asyncio.Queue()
        conversation_queue = asyncio.Queue()
        session_controller = SessionController()
        self.assertTrue(await session_controller.try_start_welcome())
        self.assertEqual(session_controller.state, EdgeSessionState.WELCOMING)
        coordinator = InteractionCoordinator(
            vision_config=VisionConfig(
                detector="simulated",
                person_threshold=0.5,
                stable_frames=1,
                welcome_cooldown_seconds=30,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=session_controller,
            disarm_listening=lambda: None,
            idle_return_to_center_seconds=0,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            await vision_queue.put(
                VisionEvent(event_type=VisionEventType.PERSON_ABSENT, confidence=0)
            )
            await asyncio.sleep(0.01)

            self.assertEqual(session_controller.state, EdgeSessionState.WELCOMING)
            self.assertEqual(action_queue.qsize(), 0)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_short_absence_is_cancelled_before_session_disengages(self) -> None:
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        action_queue = asyncio.Queue()
        conversation_queue = asyncio.Queue()
        session_controller = SessionController()
        self.assertTrue(await session_controller.try_start_welcome())
        await session_controller.recover_to_tracking("test_tracking")
        self.assertEqual(session_controller.state, EdgeSessionState.TRACKING)
        coordinator = InteractionCoordinator(
            vision_config=VisionConfig(
                detector="simulated",
                person_threshold=0.5,
                stable_frames=1,
                welcome_cooldown_seconds=30,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=session_controller,
            disarm_listening=lambda: None,
            idle_return_to_center_seconds=0.05,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            await vision_queue.put(
                VisionEvent(event_type=VisionEventType.PERSON_ABSENT, confidence=0)
            )
            await asyncio.sleep(0.01)
            self.assertEqual(session_controller.state, EdgeSessionState.TRACKING)

            await vision_queue.put(
                VisionEvent(event_type=VisionEventType.PERSON_PRESENT, confidence=1)
            )
            await asyncio.sleep(0.08)

            self.assertEqual(session_controller.state, EdgeSessionState.TRACKING)
            self.assertEqual(action_queue.qsize(), 0)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_expected_session_suppression_does_not_emit_info_logs(self) -> None:
        session_controller = SessionController()
        self.assertTrue(await session_controller.try_start_welcome())
        await session_controller.recover_to_tracking("test_tracking")
        await session_controller.note_playback_started()

        records: list[logging.LogRecord] = []
        handler = _ListHandler(records)
        logger = logging.getLogger("ai_robot_edge.session")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            self.assertFalse(await session_controller.try_start_welcome())
            self.assertFalse(await session_controller.note_person_absent())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        self.assertEqual(records, [])

    async def test_person_absence_does_not_end_active_listening_states(self) -> None:
        session_controller = SessionController()
        self.assertTrue(await session_controller.try_start_welcome())
        await session_controller.note_welcome_playback_finished()
        self.assertEqual(session_controller.state, EdgeSessionState.LISTENING)

        self.assertFalse(await session_controller.note_person_absent())
        self.assertEqual(session_controller.state, EdgeSessionState.LISTENING)

        await session_controller.note_followup_listening()
        self.assertEqual(session_controller.state, EdgeSessionState.FOLLOWUP_LISTENING)

        self.assertFalse(await session_controller.note_person_absent())
        self.assertEqual(session_controller.state, EdgeSessionState.FOLLOWUP_LISTENING)

    async def test_stable_person_queues_local_welcome_without_remote_event(self) -> None:
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        action_queue = asyncio.Queue()
        conversation_queue = asyncio.Queue()
        session_controller = SessionController()
        local_welcome_events: list[VisionEvent] = []
        listen_armed = 0

        def arm_presence_listening() -> None:
            nonlocal listen_armed
            listen_armed += 1

        async def queue_local_welcome(event: VisionEvent) -> None:
            local_welcome_events.append(event)
            await session_controller.note_welcome_playback_finished()
            arm_presence_listening()

        coordinator = InteractionCoordinator(
            vision_config=VisionConfig(
                detector="simulated",
                person_threshold=0.5,
                stable_frames=1,
                welcome_cooldown_seconds=30,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=session_controller,
            disarm_listening=lambda: None,
            idle_return_to_center_seconds=0,
            arm_presence_listening=arm_presence_listening,
            queue_local_welcome=queue_local_welcome,
            local_welcome_enabled=True,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            await vision_queue.put(
                VisionEvent(event_type=VisionEventType.PERSON_PRESENT, confidence=1)
            )
            await asyncio.sleep(0.01)

            self.assertEqual(conversation_queue.qsize(), 0)
            self.assertEqual(len(local_welcome_events), 1)
            self.assertEqual(action_queue.qsize(), 1)
            self.assertEqual(listen_armed, 1)
            self.assertEqual(session_controller.state, EdgeSessionState.LISTENING)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_person_absence_stops_presence_listening_after_delay(self) -> None:
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue()
        action_queue = asyncio.Queue()
        conversation_queue = asyncio.Queue()
        session_controller = SessionController()
        self.assertTrue(await session_controller.try_start_welcome())
        await session_controller.note_welcome_playback_finished()
        disarmed = 0

        def disarm_listening() -> None:
            nonlocal disarmed
            disarmed += 1

        coordinator = InteractionCoordinator(
            vision_config=VisionConfig(
                detector="simulated",
                person_threshold=0.5,
                stable_frames=1,
                welcome_cooldown_seconds=30,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=session_controller,
            disarm_listening=disarm_listening,
            idle_return_to_center_seconds=0.02,
            local_welcome_enabled=True,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            await vision_queue.put(
                VisionEvent(event_type=VisionEventType.PERSON_ABSENT, confidence=0)
            )
            await asyncio.sleep(0.05)

            self.assertEqual(disarmed, 1)
            self.assertEqual(session_controller.state, EdgeSessionState.DISENGAGED)
            self.assertEqual(action_queue.qsize(), 1)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class _ListHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.INFO)
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


if __name__ == "__main__":
    unittest.main()
