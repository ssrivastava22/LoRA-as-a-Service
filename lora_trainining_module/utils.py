def save_lora_adapter(model, output_dir):
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"LoRA adapter saved to {output_dir}")