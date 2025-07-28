import torch
import numpy as np

# Generate predictions from a seed sequence
def generate_from_seed(model, tokenizer, seed_sequence, max_length=50):
    # Tokenize the seed sequence
    inputs = tokenizer(seed_sequence, return_tensors="pt")
    
    # Generate continuation
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    
    # Decode the generated sequence
    generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_sequence



# Extract embeddings for supervised learning
def extract_embeddings(model, tokenizer, sequences):
    embeddings = []
    
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            inputs = tokenizer(sequence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get the last hidden state (you could also use other layers)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Average pooling of the last hidden state
            seq_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(seq_embedding)
    
    return np.array(embeddings)