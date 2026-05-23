import asyncio
import unittest

from ai_robot_edge.config import VisionConfig
from ai_robot_edge.events import ActionName, VisionEvent, VisionEventType
from ai_robot_edge.interaction import InteractionCoordinator


class WelcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_welcome_requires_stable_frames(self):
        vision_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        coordinator = InteractionCoordinator(
            VisionConfig(
                detector="simulated",
                person_threshold=0.55,
                stable_frames=2,
                welcome_cooldown_seconds=60,
            ),
            vision_queue=vision_queue,
            action_queue=action_queue,
        )
        task = asyncio.create_task(coordinator.run())
        try:
            await vision_queue.put(VisionEvent(VisionEventType.PERSON_PRESENT, 0.8))
            await asyncio.sleep(0.01)
            self.assertTrue(action_queue.empty())

            await vision_queue.put(VisionEvent(VisionEventType.PERSON_PRESENT, 0.8))
            intent = await asyncio.wait_for(action_queue.get(), timeout=1)
            self.assertEqual(intent.name, ActionName.WELCOME_MOTION)
        finally:
            task.cancel()
