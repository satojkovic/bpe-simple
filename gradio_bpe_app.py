#!/usr/bin/env python3
"""
Gradio BPE Tokenization Demo App

Interactive web interface for demonstrating Byte Pair Encoding tokenization
using trained BPE models on user-provided text.
"""

import gradio as gr
import json
import random

def get_pair_stats(tokens):
    """Calculate frequency of adjacent token pairs"""
    pair_counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

def merge_tokens(tokens, target_pair, new_token_id):
    """Replace all occurrences of target_pair with new_token_id"""
    result = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens) - 1 and 
            tokens[i] == target_pair[0] and 
            tokens[i + 1] == target_pair[1]):
            result.append(new_token_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def train_bpe(text, vocab_size=300):
    """Train BPE model"""
    tokens = list(text.encode('utf-8'))
    merges = {}
    next_token_id = 256
    num_merges = vocab_size - 256
    
    for merge_step in range(num_merges):
        pair_stats = get_pair_stats(tokens)
        
        if not pair_stats:
            break
            
        most_frequent_pair = max(pair_stats.items(), key=lambda x: x[1])
        pair, frequency = most_frequent_pair
        
        tokens = merge_tokens(tokens, pair, next_token_id)
        merges[pair] = next_token_id
        next_token_id += 1
    
    return merges, tokens

def create_vocabulary(merges):
    """Build vocabulary from merge rules"""
    vocab = {}
    
    for i in range(256):
        vocab[i] = bytes([i])
    
    for (token1, token2), merged_id in merges.items():
        vocab[merged_id] = vocab[token1] + vocab[token2]
    
    return vocab

def encode_text(text, merges):
    """Encode text using BPE"""
    tokens = list(text.encode('utf-8'))
    
    while len(tokens) >= 2:
        pair_stats = get_pair_stats(tokens)
        valid_pairs = [pair for pair in pair_stats if pair in merges]
        
        if not valid_pairs:
            break
            
        best_pair = min(valid_pairs, key=lambda p: merges[p])
        merge_id = merges[best_pair]
        tokens = merge_tokens(tokens, best_pair, merge_id)
    
    return tokens

def decode_tokens(tokens, vocab):
    """Decode token IDs back to text"""
    try:
        byte_sequence = b''.join(vocab[token_id] for token_id in tokens)
        return byte_sequence.decode('utf-8', errors='replace')
    except KeyError as e:
        return f"Error: Unknown token ID {e}"

def generate_colors(num_colors):
    """Generate distinct colors for visualization"""
    # Predefined vibrant colors with good readability
    base_colors = [
        '#FFE4B5',  # Moccasin
        '#FFB6C1',  # Light Pink
        '#B0E0E6',  # Powder Blue
        '#98FB98',  # Pale Green
        '#F0E68C',  # Khaki
        '#DDA0DD',  # Plum
        '#FFA07A',  # Light Salmon
        '#87CEEB',  # Sky Blue
        '#D2B48C',  # Tan
        '#F5DEB3',  # Wheat
        '#ADD8E6',  # Light Blue
        '#90EE90',  # Light Green
        '#FFE4E1',  # Misty Rose
        '#E0E0E0',  # Light Gray
        '#FAFAD2',  # Light Goldenrod Yellow
        '#FFCCCB',  # Light Red
        '#C8A2C8',  # Lilac
        '#BFEFFF',  # Light Sky Blue
        '#F0FFF0',  # Honeydew
        '#FDF5E6',  # Old Lace
    ]
    
    colors = []
    for i in range(num_colors):
        if i < len(base_colors):
            colors.append(base_colors[i])
        else:
            # For more colors, generate with better contrast
            hue = (i * 137.508) % 360
            saturation = 60 + (i % 4) * 10  # 60-90% saturation
            lightness = 75 + (i % 3) * 5   # 75-85% lightness
            colors.append(f"hsl({hue:.0f}, {saturation}%, {lightness}%)")
    return colors

def create_token_visualization(text, tokens, vocab):
    """Create HTML visualization of tokenized text"""
    if not tokens:
        return "<p>No tokens to display</p>"
    
    # Generate colors for each token
    colors = generate_colors(len(tokens))
    
    html_parts = []
    html_parts.append('<div style="font-family: monospace; font-size: 16px; line-height: 1.8; margin: 10px 0;">')
    
    # Track position in original text
    current_pos = 0
    original_bytes = text.encode('utf-8')
    
    for i, token_id in enumerate(tokens):
        if token_id in vocab:
            token_bytes = vocab[token_id]
            try:
                token_text = token_bytes.decode('utf-8', errors='replace')
                # Escape HTML special characters
                token_text_escaped = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                # Create colored span with tooltip
                color = colors[i % len(colors)]
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 3px 5px; margin: 1px; '
                    f'border-radius: 4px; border: 2px solid rgba(0,0,0,0.3); '
                    f'font-weight: 500; display: inline-block;" '
                    f'title="Token {i+1}: ID={token_id}, Bytes={list(token_bytes)}">'
                    f'{token_text_escaped}'
                    f'</span>'
                )
                current_pos += len(token_bytes)
            except Exception:
                # Handle non-decodable bytes
                color = colors[i % len(colors)]
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 3px 5px; margin: 1px; '
                    f'border-radius: 4px; border: 2px solid rgba(0,0,0,0.3); '
                    f'font-weight: 500; display: inline-block;" '
                    f'title="Token {i+1}: ID={token_id}, Bytes={list(token_bytes)}">'
                    f'[bytes: {list(token_bytes)}]'
                    f'</span>'
                )
    
    html_parts.append('</div>')
    
    # Add legend
    html_parts.append('<div style="margin-top: 15px; font-size: 12px; color: #666;">')
    html_parts.append('<strong>Legend:</strong> Each colored block represents one BPE token. Hover over tokens to see details.')
    html_parts.append('</div>')
    
    return ''.join(html_parts)

# Global variables for trained model
TRAINING_TEXT = "こんにちは世界！Hello world! 機械学習は面白いですね。Machine learning is fascinating! Programming プログラミング 人工知能 AI technology テクノロジー"
MERGES, _ = train_bpe(TRAINING_TEXT, vocab_size=350)
VOCAB = create_vocabulary(MERGES)

def tokenize_text(input_text):
    """Main function for tokenizing user input"""
    if not input_text.strip():
        return "Please enter some text to tokenize.", "", "", "", ""
    
    try:
        # Encode using trained BPE
        encoded_tokens = encode_text(input_text, MERGES)
        
        # Create detailed analysis
        byte_tokens = list(input_text.encode('utf-8'))
        
        # Decode to verify
        decoded_text = decode_tokens(encoded_tokens, VOCAB)
        
        # Create visualization
        visualization_html = create_token_visualization(input_text, encoded_tokens, VOCAB)
        
        # Format results
        token_info = f"**Original text:** {input_text}\n\n"
        token_info += f"**UTF-8 bytes:** {len(byte_tokens)} tokens\n"
        token_info += f"**BPE tokens:** {len(encoded_tokens)} tokens\n"
        token_info += f"**Compression ratio:** {len(byte_tokens) / len(encoded_tokens):.2f}x\n\n"
        token_info += f"**Decoded verification:** {decoded_text}\n"
        token_info += f"**Accuracy:** {'✓ Perfect match' if input_text == decoded_text else '✗ Mismatch detected'}"
        
        # Token details
        token_details = "**BPE Token IDs:**\n" + str(encoded_tokens) + "\n\n"
        token_details += "**Token Breakdown:**\n"
        for i, token_id in enumerate(encoded_tokens):
            if token_id in VOCAB:
                token_bytes = VOCAB[token_id]
                try:
                    token_str = token_bytes.decode('utf-8', errors='replace')
                    token_details += f"Token {i+1}: ID={token_id} → '{token_str}' (bytes: {list(token_bytes)})\n"
                except:
                    token_details += f"Token {i+1}: ID={token_id} → bytes: {list(token_bytes)}\n"
        
        # UTF-8 comparison
        utf8_details = "**UTF-8 Byte Sequence:**\n" + str(byte_tokens) + "\n\n"
        utf8_details += "**Byte Breakdown:**\n"
        for i, byte_val in enumerate(byte_tokens):
            char = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
            utf8_details += f"Byte {i+1}: {byte_val} → '{char}'\n"
        
        return token_info, token_details, utf8_details, visualization_html, ""
        
    except Exception as e:
        return f"Error processing text: {str(e)}", "", "", "", ""

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="BPE Tokenization Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔤 Byte Pair Encoding (BPE) Tokenization Demo
        
        This app demonstrates how Byte Pair Encoding works by tokenizing your input text.
        The model has been pre-trained on multilingual text (Japanese and English).
        
        **Enter any text below to see how it gets tokenized!**
        """)
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to tokenize (supports Japanese, English, and other languages)",
                    lines=3,
                    value="こんにちは Hello 機械学習"
                )
                
                tokenize_btn = gr.Button("🚀 Tokenize Text", variant="primary")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(label="Summary")
        
        with gr.Row():
            with gr.Column():
                visualization_output = gr.HTML(label="Token Visualization")
            
        with gr.Row():
            with gr.Column():
                token_output = gr.Markdown(label="BPE Token Details")
            with gr.Column():
                utf8_output = gr.Markdown(label="UTF-8 Byte Details")
        
        with gr.Row():
            error_output = gr.Textbox(label="Errors", visible=False)
        
        # Examples
        gr.Examples(
            examples=[
                ["Hello world!"],
                ["こんにちは"],
                ["機械学習は面白い"],
                ["Programming プログラミング"],
                ["人工知能 AI technology"],
                ["🤖 Unicode symbols ✨"],
                ["Mix 混合 text テキスト!"]
            ],
            inputs=input_text
        )
        
        # Event handlers
        tokenize_btn.click(
            fn=tokenize_text,
            inputs=[input_text],
            outputs=[summary_output, token_output, utf8_output, visualization_output, error_output]
        )
        
        input_text.submit(
            fn=tokenize_text,
            inputs=[input_text],
            outputs=[summary_output, token_output, utf8_output, visualization_output, error_output]
        )
        
        gr.Markdown("""
        ---
        
        ### 📚 About BPE
        
        **Byte Pair Encoding (BPE)** is a subword tokenization algorithm used in modern language models like GPT.
        
        - **Training**: Iteratively merges the most frequent adjacent byte pairs
        - **Compression**: Reduces token count while preserving information
        - **Multilingual**: Works with any UTF-8 text including emojis and special characters
        - **Subword units**: Handles out-of-vocabulary words by breaking them into smaller pieces
        
        This demo uses a model trained on multilingual text with a vocabulary size of 350 tokens.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    print("Starting Gradio app...")
    demo.launch(share=False, debug=False, server_port=7860)