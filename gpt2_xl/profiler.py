import jax

class JAXProfileManager:
    """
    Context manager to profile a JAX execution block using jax.profiler.trace.
    """
    def __init__(self, trace_dir: str, create_perfetto_link: bool = False):
        self.trace_dir = trace_dir
        self.create_perfetto_link = create_perfetto_link
        self._context = None

    def __enter__(self):
        print("[JAXProfileManager] Starting profiling with trace_dir:", self.trace_dir)
        self._context = jax.profiler.trace(self.trace_dir, create_perfetto_link=self.create_perfetto_link)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._context.__exit__(exc_type, exc_val, exc_tb)
        print("[JAXProfileManager] Profiling finished.")
        return result
