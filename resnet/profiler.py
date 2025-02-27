import jax

class JAXProfileManager:

    def __init__(self, trace_dir: str, create_perfetto_link: bool = False):
        self.trace_dir = trace_dir
        self.create_perfetto_link = create_perfetto_link
        self._context = None

    def __enter__(self):
        self._context = jax.profiler.trace(self.trace_dir, create_perfetto_link=self.create_perfetto_link)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._context.__exit__(exc_type, exc_val, exc_tb)
