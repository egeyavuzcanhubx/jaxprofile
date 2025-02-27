from gpt_model import GPT2Model
from profiler import JAXProfileManager

def main():
    print("[profiler_example] Starting profiler_example.py")
    prompt = "The Manhattan bridge"
    trace_dir = "/home/egeyavuzcan/jax/flaxmodels/flaxmodels/gpt2/profiler_outputs"  # Update path as needed

    print("[profiler_example] Initializing GPT2Model...")
    model = GPT2Model(prompt=prompt, model_name="gpt2-xl", num_tokens=3)

    print("[profiler_example] Running generation within profiling context...")
    with JAXProfileManager(trace_dir):
        generated_text = model.generate()

    print("[profiler_example] Generation finished. Output:")
    print(generated_text)

if __name__ == "__main__":
    main()
