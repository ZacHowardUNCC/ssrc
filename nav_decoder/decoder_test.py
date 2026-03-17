import torch

from nav_decoder.decoder_model import SimpleNavDecoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load decoder
    decoder = SimpleNavDecoder().to(device)
    decoder.load_state_dict(
        torch.load(
            "/home/charml/Documents/UniVLA-nav/jetson/ros2_ws/src/nav_decoder/weights/decoder_nav_v1.pt",
            map_location=device,
        )
    )
    decoder.eval()

    # Create fake inputs with correct shapes
    visual_embed = torch.randn(1, 256, 4096, device=device, dtype=torch.float32)
    latent_action = torch.randn(1, 4, 4096, device=device, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        logits = decoder(visual_embed, latent_action)
        action_id = logits.argmax(dim=1).item()

    print("Predicted action id:", action_id)


if __name__ == "__main__":
    main()
