"""
Lesson-Based Curriculum Scheduling

Implements automatic curriculum progression based on performance metrics.

Usage:
    scheduler = CurriculumScheduler(lessons)
    if scheduler.should_progress(reward_buffer, progress):
        scheduler.advance_lesson()
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from mlagents_envs import logging_util
from mlagents.trainers.settings import (
    Lesson,
    CompletionCriteriaSettings,
)

logger = logging_util.get_logger(__name__)


@dataclass
class CurriculumProgress:
    """Tracks curriculum progress"""
    current_lesson: int
    lessons_completed: int
    total_lessons: int
    smoothed_reward: float
    steps_in_lesson: int


class CurriculumScheduler:
    """
    Automatic lesson progression for curriculum learning
    
    Features:
    - Progress-based advancement
    - Reward-based advancement
    - Smoothed metrics for stability
    - Minimum lesson duration
    - Automatic parameter updates
    """
    
    def __init__(
        self,
        lessons: List[Lesson],
        parameter_name: str,
        min_lesson_steps: int = 10000
    ):
        """
        Initialize curriculum scheduler
        
        :param lessons: List of curriculum lessons
        :param parameter_name: Name of environment parameter to control
        :param min_lesson_steps: Minimum steps before considering advancement
        """
        self.lessons = lessons
        self.parameter_name = parameter_name
        self.min_lesson_steps = min_lesson_steps
        
        self.current_lesson_index = 0
        self.steps_in_current_lesson = 0
        self.smoothed_reward = 0.0
        self.lesson_history: List[Dict] = []
        
        logger.info(
            f"Initialized curriculum scheduler for '{parameter_name}' "
            f"with {len(lessons)} lessons"
        )
    
    @property
    def current_lesson(self) -> Lesson:
        """Get current lesson"""
        return self.lessons[self.current_lesson_index]
    
    @property
    def is_final_lesson(self) -> bool:
        """Check if on final lesson"""
        return self.current_lesson_index == len(self.lessons) - 1
    
    @property
    def progress_percentage(self) -> float:
        """Get curriculum progress as percentage"""
        return (self.current_lesson_index / len(self.lessons)) * 100
    
    def get_current_parameters(self) -> Dict:
        """Get current lesson parameters"""
        return {
            self.parameter_name: self.current_lesson.value
        }
    
    def should_progress(
        self,
        reward_buffer: List[float],
        training_progress: float
    ) -> Tuple[bool, str]:
        """
        Check if should advance to next lesson
        
        :param reward_buffer: Recent episode rewards
        :param training_progress: Overall training progress (0-1)
        :return: (should_advance, reason)
        """
        self.steps_in_current_lesson += 1
        
        # Already on final lesson
        if self.is_final_lesson:
            return (False, "Final lesson")
        
        # Check minimum steps requirement
        if self.steps_in_current_lesson < self.min_lesson_steps:
            return (False, f"Only {self.steps_in_current_lesson}/{self.min_lesson_steps} steps")
        
        # Get completion criteria
        criteria = self.current_lesson.completion_criteria
        if criteria is None:
            return (False, "No completion criteria")
        
        # Check criteria
        should_advance, new_smoothing = criteria.need_increment(
            training_progress,
            reward_buffer,
            self.smoothed_reward
        )
        
        self.smoothed_reward = new_smoothing
        
        if should_advance:
            measure_type = criteria.measure.value
            threshold = criteria.threshold
            
            if measure_type == "progress":
                reason = f"Progress {training_progress:.2%} > {threshold:.2%}"
            else:  # reward
                mean_reward = np.mean(reward_buffer) if reward_buffer else 0
                reason = f"Reward {mean_reward:.2f} > {threshold:.2f}"
            
            return (True, reason)
        
        return (False, "Criteria not met")
    
    def advance_lesson(self) -> bool:
        """
        Advance to next lesson
        
        :return: True if advanced, False if already on final lesson
        """
        if self.is_final_lesson:
            logger.warning("Already on final lesson, cannot advance")
            return False
        
        # Record lesson completion
        lesson_stats = {
            "lesson_index": self.current_lesson_index,
            "lesson_name": self.current_lesson.name,
            "steps": self.steps_in_current_lesson,
            "final_smoothed_reward": self.smoothed_reward,
        }
        self.lesson_history.append(lesson_stats)
        
        # Advance
        self.current_lesson_index += 1
        self.steps_in_current_lesson = 0
        
        logger.info(
            f"Advanced to lesson {self.current_lesson_index + 1}/{len(self.lessons)}: "
            f"{self.current_lesson.name}"
        )
        
        return True
    
    def get_progress(self) -> CurriculumProgress:
        """Get current progress snapshot"""
        return CurriculumProgress(
            current_lesson=self.current_lesson_index,
            lessons_completed=len(self.lesson_history),
            total_lessons=len(self.lessons),
            smoothed_reward=self.smoothed_reward,
            steps_in_lesson=self.steps_in_current_lesson
        )
    
    def reset(self):
        """Reset to first lesson"""
        self.current_lesson_index = 0
        self.steps_in_current_lesson = 0
        self.smoothed_reward = 0.0
        logger.info("Reset curriculum to first lesson")


class MultiParameterCurriculumScheduler:
    """
    Manages multiple curriculum parameters simultaneously
    
    Use when you need to schedule multiple environment parameters
    with independent progression criteria.
    """
    
    def __init__(self):
        self.schedulers: Dict[str, CurriculumScheduler] = {}
    
    def add_scheduler(
        self,
        parameter_name: str,
        scheduler: CurriculumScheduler
    ):
        """Add a scheduler for a parameter"""
        self.schedulers[parameter_name] = scheduler
        logger.info(f"Added scheduler for parameter: {parameter_name}")
    
    def get_current_parameters(self) -> Dict:
        """Get all current parameters"""
        params = {}
        for param_name, scheduler in self.schedulers.items():
            params.update(scheduler.get_current_parameters())
        return params
    
    def update_all(
        self,
        reward_buffer: List[float],
        training_progress: float
    ) -> List[str]:
        """
        Update all schedulers
        
        :return: List of parameters that advanced
        """
        advanced = []
        
        for param_name, scheduler in self.schedulers.items():
            should_advance, reason = scheduler.should_progress(
                reward_buffer,
                training_progress
            )
            
            if should_advance:
                if scheduler.advance_lesson():
                    advanced.append(param_name)
                    logger.info(
                        f"Advanced curriculum parameter '{param_name}': {reason}"
                    )
        
        return advanced
    
    def get_all_progress(self) -> Dict[str, CurriculumProgress]:
        """Get progress for all parameters"""
        return {
            param_name: scheduler.get_progress()
            for param_name, scheduler in self.schedulers.items()
        }
