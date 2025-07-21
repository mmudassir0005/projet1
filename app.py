import numpy as np
from PIL import Image
import wave
import os
import cv2
import io
from flask import Flask, request, send_file, jsonify, render_template

app = Flask(__name__)

END_MARKER = '1111111111111110'

def text_to_bin(text):
    return ''.join(format(ord(i), '08b') for i in text)

def bin_to_text(bin_str):
    chars = []
    for i in range(0, len(bin_str), 8):
        byte = bin_str[i:i+8]
        if len(byte) < 8:  # Ignore incomplete byte.
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

# --------- EMBED FUNCTIONS ---------

def embed_text_in_image(image, text):
    binary_text = text_to_bin(text) + END_MARKER
    img = Image.open(image)
    img = img.convert('RGBA')
    data = np.array(img)
    flat_data = data.flatten()
    if len(binary_text) > len(flat_data):
        raise ValueError('Text too long for this image')
    for i in range(len(binary_text)):
        flat_data[i] = (flat_data[i] & 254) | int(binary_text[i])
    new_data = flat_data.reshape(data.shape)
    new_img = Image.fromarray(new_data.astype('uint8'), 'RGBA')
    buffer = io.BytesIO()
    new_img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def embed_text_in_audio(audio_file, text):
    binary_text = text_to_bin(text) + END_MARKER
    with wave.open(audio_file, 'rb') as wav_in:
        params = wav_in.getparams()
        frames = bytearray(wav_in.readframes(wav_in.getnframes()))
    if len(binary_text) > len(frames):
        raise ValueError('Text too long for this audio')
    for i in range(len(binary_text)):
        frames[i] = (frames[i] & 254) | int(binary_text[i])
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(bytes(frames))
    buffer.seek(0)
    return buffer

def embed_text_in_video(video_file, text):
    binary_text = text_to_bin(text) + END_MARKER
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError('Could not open video')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = 'temp_stego.avi'
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    bit_idx = 0
    max_bits = width * height * 3 * frame_count
    
    if len(binary_text) > max_bits:
        cap.release()
        out.release()
        raise ValueError('Text too long for this video')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flat_frame = frame.flatten()
        for i in range(len(flat_frame)):
            if bit_idx < len(binary_text):
                flat_frame[i] = (flat_frame[i] & 254) | int(binary_text[bit_idx])
                bit_idx += 1
            else:
                break
        frame = flat_frame.reshape(frame.shape)
        out.write(frame)
    cap.release()
    out.release()
    with open(temp_output, 'rb') as f:
        video_data = f.read()
    os.remove(temp_output)
    return io.BytesIO(video_data)

# --------- EXTRACT FUNCTIONS ---------

def extract_text_from_image(image, END_MARKER='1111111111111110'):
    from PIL import Image
    import numpy as np

    def bin_to_text(bin_str):
        chars = []
        for i in range(0, len(bin_str), 8):
            byte = bin_str[i:i+8]
            if len(byte) < 8:
                break
            chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    img = Image.open(image)
    img = img.convert('RGBA')
    data = np.array(img)
    flat_data = data.flatten()
    bits = []
    marker_len = len(END_MARKER)
    # Efficiently process until marker is found
    for val in flat_data:
        bits.append(str(val & 1))
        if len(bits) >= marker_len:
            recent_bits = ''.join(bits[-marker_len*2:])  # Check recent window for marker
            idx = recent_bits.find(END_MARKER)
            if idx != -1:
                end_index = len(bits) - len(recent_bits) + idx
                bin_text = ''.join(bits[:end_index])
                return bin_to_text(bin_text)
    return ''

def extract_text_from_audio(audio_file, END_MARKER='1111111111111110'):
    import wave

    def bin_to_text(bin_str):
        chars = []
        for i in range(0, len(bin_str), 8):
            byte = bin_str[i:i+8]
            if len(byte) < 8:
                break
            chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    bits = []
    marker_len = len(END_MARKER)

    with wave.open(audio_file, 'rb') as wav_in:
        frames = bytearray(wav_in.readframes(wav_in.getnframes()))

    for byte in frames:
        bits.append(str(byte & 1))
        # Only check for marker after each full byte is formed
        if len(bits) % 8 == 0 and len(bits) >= marker_len:
            # Only scan a recent window for the marker for speed
            recent_bits = ''.join(bits[-marker_len * 2:])
            idx = recent_bits.find(END_MARKER)
            if idx != -1:
                bit_string = ''.join(bits)
                end_idx = bit_string.find(END_MARKER)
                content_bits = bit_string[:end_idx]
                return bin_to_text(content_bits)
    return ''  # End marker not found


def extract_text_from_video(video_file, END_MARKER='1111111111111110'):
    import cv2

    def bin_to_text(bin_str):
        chars = []
        for i in range(0, len(bin_str), 8):
            byte = bin_str[i:i+8]
            if len(byte) < 8:
                break
            chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    cap = cv2.VideoCapture(video_file)
    bits = []
    marker_len = len(END_MARKER)
    found = False
    # Use a running window to check for the marker efficiently
    for_frame = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flat_frame = frame.flatten()
        for pixel in flat_frame:
            bits.append(str(pixel & 1))
            # Only check for marker after each byte
            if len(bits) >= marker_len and len(bits) % 8 == 0:
                # For efficiency, check only the last chunk of bits
                recent_bits = ''.join(bits[-marker_len-8:])  # Check last bytes plus marker size
                marker_idx = recent_bits.find(END_MARKER)
                if marker_idx != -1:
                    # Calculate position in full bits
                    abs_idx = len(bits) - len(recent_bits) + marker_idx
                    bit_str = ''.join(bits[:abs_idx])
                    cap.release()
                    return bin_to_text(bit_str)
    cap.release()
    return ''


# --------- FRONTEND ROUTES ---------
@app.route('/')
@app.route('/encrypt')
def encrypt():
    return render_template('encrypt.html')

@app.route('/decrypt')
def decrypt():
    return render_template('decrypt.html')

@app.route('/hide', methods=['POST'])
def hide_text():
    media_type = request.form.get('mediaType')
    text_message = request.form.get('textMessage')
    media_file = request.files.get('mediaFile')
    if not media_file or not text_message or not media_type:
        return 'Missing input', 400
    try:
        if media_type == 'image':
            if not media_file.filename.lower().endswith('.png'):
                return 'Only PNG images are supported', 400
            result_file = embed_text_in_image(media_file, text_message)
            return send_file(result_file, mimetype='image/png', download_name='stego.png')
        elif media_type == 'audio':
            if not media_file.filename.lower().endswith('.wav'):
                return 'Only WAV audio files are supported', 400
            result_file = embed_text_in_audio(media_file, text_message)
            return send_file(result_file, mimetype='audio/wav', download_name='stego.wav')
        elif media_type == 'video':
            if not media_file.filename.lower().endswith('.avi'):
                return 'Only AVI video files are supported', 400
            result_file = embed_text_in_video(media_file, text_message)
            return send_file(result_file, mimetype='video/x-msvideo', download_name='stego.avi')
        else:
            return 'Invalid media type', 400
    except Exception as e:
        return str(e), 500

@app.route('/extract', methods=['POST'])
def extract_text():
    media_type = request.form.get('extractMediaType')
    media_file = request.files.get('extractMediaFile')
    if not media_file or not media_type:
        return 'Missing input', 400
    try:
        if media_type == 'image':
            if not media_file.filename.lower().endswith('.png'):
                return 'Only PNG images are supported', 400
            text_message = extract_text_from_image(media_file)
        elif media_type == 'audio':
            if not media_file.filename.lower().endswith('.wav'):
                return 'Only WAV audio files are supported', 400
            media_file.seek(0)
            text_message = extract_text_from_audio(media_file)
        elif media_type == 'video':
            if not media_file.filename.lower().endswith('.avi'):
                return 'Only AVI video files are supported', 400
            media_file.seek(0)
            temp_video = 'temp_extr.avi'
            with open(temp_video, 'wb') as f:
                f.write(media_file.read())
            text_message = extract_text_from_video(temp_video)
            os.remove(temp_video)
        else:
            return 'Invalid media type', 400
        return jsonify({'text_message': text_message})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
