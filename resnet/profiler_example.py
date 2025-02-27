from resnet_model import ResNetModel
from profiler import JAXProfileManager

def main():
    # Güncellemeniz gereken yollar:
    image_path = "/home/egeyavuzcan/jax/flaxmodels/flaxmodels/resnet/dimg-18-weighted.png"
    trace_dir = "/home/egeyavuzcan/jax/flaxmodels/flaxmodels/resnet/profile/jax-trace-resnet"

    # Modeli initialize edelim.
    model = ResNetModel(image_path)
    model.load_image()
    model.initialize_model()
    model.warmup(steps=2)

    # Profil işlemini context manager kullanarak yapıyoruz.
    with JAXProfileManager(trace_dir):
        logits = model.inference()

    print("[main] Profil işlemi tamamlandı. Inference çıktısının şekli:", logits.shape)

if __name__ == "__main__":
    main()
