from resnet_model import ResNetModel
from profiler import JAXProfileManager

def main():
    
    image_path = "/home/egeyavuzcan/jax/flaxmodels/flaxmodels/resnet/dimg-18-weighted.png"
    trace_dir = "/home/egeyavuzcan/jax_profile/jaxprofile/resnet"

    
    model = ResNetModel(image_path)
    model.load_image()
    model.initialize_model()
    model.warmup(steps=2)

   
    with JAXProfileManager(trace_dir):
        logits = model.inference()

    print("[main] Profile finished", logits.shape)

if __name__ == "__main__":
    main()
