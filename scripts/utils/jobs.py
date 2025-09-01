import threading
from collections import deque
from typing import Any, Callable


class JobsQueueExecutor:
    def __init__(self, capacity: int | None = 64):
        self._condition = threading.Condition()
        self._jobs_queue = deque(maxlen=capacity)
        self._closed_ref = [False]
        self._error_ref = [None]
        self._thread = threading.Thread(
            target=self.thread_func, daemon=True,
            args=(self._condition, self._jobs_queue, self._closed_ref, self._error_ref))
        self._thread.start()

    @classmethod
    def thread_func(cls,
        condition: threading.Condition,
        jobs_queue: deque,
        closed_ref: list[bool],
        error_ref: list[Exception | None],
    ):
        while True:
            with condition:
                condition.wait_for(lambda: closed_ref[0] or len(jobs_queue) > 0)
                if len(jobs_queue) > 0:
                    job = jobs_queue.popleft()
                    condition.notify_all()
                elif closed_ref[0]:
                    return

            try:
                job()
            except Exception as err:
                with condition:
                    closed_ref[0] = True
                    error_ref[0] = err
                    jobs_queue.clear()
                    condition.notify_all()

    def append(self, job: Callable[[], Any]):
        condition = self._condition
        jobs_queue = self._jobs_queue
        closed_ref = self._closed_ref
        with condition:
            condition.wait_for(lambda: closed_ref[0] or len(jobs_queue) < jobs_queue.maxlen)
            if closed_ref[0]:
                raise RuntimeError("JobsQueueExecutor has been closed!")
            if len(jobs_queue) < jobs_queue.maxlen:
                jobs_queue.append(job)
                condition.notify_all()

    def close(self, cancel: bool = False):
        condition = self._condition
        jobs_queue = self._jobs_queue
        closed_ref = self._closed_ref
        with condition:
            if cancel:
                jobs_queue.clear()
            closed_ref[0] = True
            condition.notify_all()

    def join(self):
        self._thread.join()
        if error := self._error_ref[0]:
            self._error_ref[0] = None
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.join()

    def __del__(self):
        self.close()
        self.join()
