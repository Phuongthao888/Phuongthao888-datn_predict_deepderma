import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không có GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Input, Dropout, Dense, Layer, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================================================
# CẤU HÌNH MÔ HÌNH
# ==============================================================

# CẬP NHẬT THEO KẾT QUẢ TUNING VÀ PHÙ HỢP VỚI ẢNH 32x32
NUM_LAYERS = 6         
NUM_HEADS = 12         
D = 416                
MLP_DIM = 1536         
PATCH_SIZE = 8         
IMAGE_SIZE = 32        
DROPOUT = 0.2          
NORM_EPS = 1e-12

# ==============================================================
# CÁC LỚP CUSTOM CHO ViT
# ==============================================================

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        dim = patches.shape[-1]
        patches = tf.reshape(patches, (batch_size, -1, dim))

        return patches

class PatchEmbedding(Layer):
    def __init__(self, patch_size, image_size, projection_dim):
        super(PatchEmbedding, self).__init__()

        self.num_patches = (image_size // patch_size) ** 2

        self.cls_token = self.add_weight(
            name="cls_token",
            shape=[1, 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

        self.patches = Patches(patch_size)
        self.projection = Dense(units=projection_dim)

        self.position_embdding = self.add_weight(
            name="position_embeddings",
            shape=[self.num_patches + 1, projection_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

    def call(self, images):
        patch = self.patches(images)
        encoded_patches = self.projection(patch)

        batch_size = tf.shape(images)[0]
        hidden_size = tf.shape(encoded_patches)[-1]

        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
            dtype=images.dtype
        )

        encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)
        encoded_patches += self.position_embdding

        return encoded_patches

class MLPBlock(Layer):
    def __init__(self, hidden_layers, dropout=0.1, activation='gelu'):
        super(MLPBlock, self).__init__()

        layers = []
        for num_units in hidden_layers:
            layers.extend([
                Dense(num_units, activation=activation),
                Dropout(dropout)
            ])

        self.mlp = Sequential(layers)

    def call(self, inputs):
        outputs = self.mlp(inputs)
        return outputs

class TransformerBlock(Layer):
    def __init__(self, num_heads, D, hidden_layers, dropout=0.1, norm_eps=1e-12):
        super(TransformerBlock, self).__init__()

        self.norm = LayerNormalization(epsilon=norm_eps)
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=D // num_heads, dropout=dropout
        )
        self.mlp = MLPBlock(hidden_layers, dropout)

    def call(self, inputs):
        norm_attention = self.norm(inputs)
        attention = self.attention(query=norm_attention, value=norm_attention)
        attention += inputs
        outputs = self.mlp(self.norm(attention))
        outputs += attention

        return outputs

class TransformerEncoder(Layer):
    def __init__(self, num_layers, num_heads, D, mlp_dim, dropout=0.1, norm_eps=1e-12):
        super(TransformerEncoder, self).__init__()

        transformer_blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(
                num_heads=num_heads,
                D=D,
                hidden_layers=[mlp_dim, D],
                dropout=dropout,
                norm_eps=norm_eps
            )
            transformer_blocks.append(block)
        self.encoder = Sequential(transformer_blocks)

    def call(self, inputs):
        outputs = self.encoder(inputs)
        return outputs

class ViT(Model):
    def __init__(self, num_classes, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, D=D, mlp_dim=MLP_DIM, 
                 patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, dropout=DROPOUT, norm_eps=NORM_EPS):
        super(ViT, self).__init__()
        
        self.embedding = PatchEmbedding(patch_size, image_size, D)
        
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            D=D,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_eps=norm_eps
        )
        
        self.mlp_head = Sequential([
            LayerNormalization(epsilon=norm_eps),
            Dense(mlp_dim),
            Dropout(dropout),
            Dense(num_classes, activation='sigmoid')
        ])
        
        self.last_layer_norm = LayerNormalization(epsilon=norm_eps)

    def call(self, inputs):
        embedded = self.embedding(inputs)
        encoded = self.encoder(embedded)
        embedded_cls = encoded[:, 0]
        y = self.last_layer_norm(embedded_cls)
        output = self.mlp_head(y)
        return output

# ==============================================================
# CÁC LỚP CHO VISUALIZATION
# ==============================================================

class ViTWithAttention(Model):
    def __init__(self, vit_model):
        super(ViTWithAttention, self).__init__()
        self.vit_model = vit_model
        
    def call(self, inputs):
        embedded = self.vit_model.embedding(inputs)
        
        attention_maps = []
        
        x = embedded
        for i, block in enumerate(self.vit_model.encoder.encoder.layers):
            norm_attention = block.norm(x)
            attention_output, attention_weights = block.attention(
                query=norm_attention, 
                value=norm_attention,
                return_attention_scores=True
            )
            attention_maps.append(attention_weights)
            
            attention_output += x
            mlp_output = block.mlp(block.norm(attention_output))
            x = mlp_output + attention_output
        
        embedded_cls = x[:, 0]
        y = self.vit_model.last_layer_norm(embedded_cls)
        output = self.vit_model.mlp_head(y)
        
        return output, attention_maps

class ViTAttentionExtractor(Model):
    def __init__(self, vit_model):
        super(ViTAttentionExtractor, self).__init__()
        self.vit_model = vit_model
        
    def call(self, inputs):
        embedded = self.vit_model.embedding(inputs)
        
        all_attention_maps = []
        
        x = embedded
        for block in self.vit_model.encoder.encoder.layers:
            norm_attention = block.norm(x)
            attention_output, attention_weights = block.attention(
                query=norm_attention, 
                value=norm_attention,
                return_attention_scores=True
            )
            all_attention_maps.append(attention_weights)
            
            attention_output += x
            mlp_output = block.mlp(block.norm(attention_output))
            x = mlp_output + attention_output
        
        embedded_cls = x[:, 0]
        y = self.vit_model.last_layer_norm(embedded_cls)
        output = self.vit_model.mlp_head(y)
        
        return output, all_attention_maps

# ==============================================================
# CÁC HÀM TIỆN ÍCH
# ==============================================================

def compute_vit_gradcam(model, img, class_idx=0, layer_idx=-1):
    img_tensor = tf.expand_dims(img, axis=0)
    
    with tf.GradientTape() as tape:
        predictions, attention_maps = model(img_tensor)
        
        if layer_idx == -1:
            layer_idx = len(attention_maps) - 1
        
        attention_weights = attention_maps[layer_idx]
        cls_attention = tf.reduce_mean(attention_weights[:, :, 0, 1:], axis=1)
        cls_attention = tf.reduce_mean(cls_attention, axis=0)
        
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, attention_maps[layer_idx])
    grads = tf.reduce_mean(grads[:, :, 0, 1:], axis=[0, 1])
    
    guided_grads = tf.cast(grads > 0, 'float32') * grads * cls_attention
    
    num_patches = int(np.sqrt(guided_grads.shape[0]))
    cam = guided_grads.numpy().reshape((num_patches, num_patches))
    
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    return cam, predictions.numpy()[0]

def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlayed

def overlay_alpha_blend(image, heatmap, color=(255, 0, 0), alpha=0.6):
    image_uint8 = (image * 255).astype('uint8')
    
    color_mask = np.zeros_like(image_uint8)
    color_mask[..., 0] = color[0]
    color_mask[..., 1] = color[1]
    color_mask[..., 2] = color[2]
    
    alpha_mask = np.expand_dims(heatmap * alpha, axis=-1)
    result = image_uint8 * (1 - alpha_mask) + color_mask * alpha_mask
    result = result.astype('uint8')
    
    return result

def visualize_attention_maps(img, attention_maps, layer_idx=-1, head_idx=0):
    if layer_idx == -1:
        layer_idx = len(attention_maps) - 1
    
    attention_weights = attention_maps[layer_idx]
    cls_attention = attention_weights[0, head_idx, 0, 1:]
    
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    attention_map_2d = cls_attention.numpy().reshape((num_patches, num_patches))
    
    attention_map_resized = cv2.resize(attention_map_2d, (img.shape[1], img.shape[0]))
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min() + 1e-8)
    
    return attention_map_resized

def fuse_gradcam_attention(gradcam_heatmap, attention_map, method='weighted_average'):
    if gradcam_heatmap.shape != attention_map.shape:
        attention_map = cv2.resize(attention_map, (gradcam_heatmap.shape[1], gradcam_heatmap.shape[0]))
    
    if method == 'weighted_average':
        fused = 0.7 * gradcam_heatmap + 0.3 * attention_map
    elif method == 'multiply':
        fused = gradcam_heatmap * attention_map
    elif method == 'max':
        fused = np.maximum(gradcam_heatmap, attention_map)
    elif method == 'attention_guided':
        attention_mask = (attention_map > 0.3).astype(float)
        fused = gradcam_heatmap * attention_mask
    
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    return fused

def attention_rollout(attention_maps, start_layer=0):
    batch_size, num_heads, num_patches, _ = attention_maps[0].shape
    attention_rollout_matrix = tf.eye(num_patches, num_patches)
    attention_rollout_matrix = tf.tile(tf.expand_dims(attention_rollout_matrix, 0), [batch_size, 1, 1])
    
    for attention_weights in attention_maps[start_layer:]:
        attention_weights_avg = tf.reduce_mean(attention_weights, axis=1)
        attention_rollout_matrix = tf.matmul(attention_weights_avg, attention_rollout_matrix)
    
    return attention_rollout_matrix

# ==============================================================
# KHỞI TẠO MÔ HÌNH VÀ FLASK APP
# ==============================================================

# Khởi tạo mô hình
img_size = IMAGE_SIZE
channel = 3

inputs = Input(shape=(img_size, img_size, channel)) 
vit_model = ViT(
    num_classes=1,
    image_size=img_size, 
    patch_size=PATCH_SIZE        
)

outputs = vit_model(inputs)  
model = Model(inputs=inputs, outputs=outputs)

# Compile model
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.0004, weight_decay=1.9091e-06
)

model.compile(
    optimizer=optimizer, 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# Load weights
model.load_weights('../training_results_deepderma_vit/deepderma_vit_version2_weights.best.weights.h5')

# Khởi tạo models với attention
vit_with_attention = ViTWithAttention(vit_model)
vit_attention_extractor = ViTAttentionExtractor(vit_model)

# Class names
class_names = ['Benign', 'Malignant']

# ==============================================================
# FLASK APPLICATION
# ==============================================================

from flask import Flask, request, render_template, jsonify, send_file
import io
import base64
from PIL import Image

app = Flask(__name__)

def preprocess_image(image):
    """Tiền xử lý ảnh đầu vào"""
    # Resize về kích thước model yêu cầu
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Convert to array và normalize
    img_array = np.array(image) / 255.0
    return img_array

def analyze_image_comprehensive(img_array):
    """Phân tích ảnh toàn diện và trả về tất cả kết quả"""
    # Dự đoán cơ bản
    img_tensor = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_tensor, verbose=0)[0][0]
    
    # Xác định class và confidence
    pred_class_idx = int(prediction > 0.5)
    pred_class = class_names[pred_class_idx]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Lấy tất cả visualization
    gradcam_heatmap, _ = compute_vit_gradcam(vit_with_attention, img_array)
    _, attention_maps = vit_attention_extractor(img_tensor)
    attention_map = visualize_attention_maps(img_array, attention_maps, layer_idx=-1, head_idx=0)
    fused_map = fuse_gradcam_attention(gradcam_heatmap, attention_map)
    
    # Lấy attention từ nhiều heads
    attention_heads = []
    for i in range(4):  # 4 heads đầu tiên
        head_map = visualize_attention_maps(img_array, attention_maps, layer_idx=-1, head_idx=i)
        attention_heads.append(head_map)
    
    # Lấy attention từ nhiều layers
    layer_attention = []
    layer_indices = [0, len(attention_maps)//3, 2*len(attention_maps)//3, -1]
    for layer_idx in layer_indices:
        layer_map = visualize_attention_maps(img_array, attention_maps, layer_idx=layer_idx, head_idx=0)
        layer_attention.append(layer_map)
    
    # Attention rollout
    rollout_matrix = attention_rollout(attention_maps)
    rollout_attention = rollout_matrix[0, 0, 1:].numpy()
    num_patches = int(np.sqrt(rollout_attention.shape[0]))
    rollout_2d = rollout_attention.reshape((num_patches, num_patches))
    rollout_resized = cv2.resize(rollout_2d, (img_array.shape[1], img_array.shape[0]))
    rollout_resized = (rollout_resized - rollout_resized.min()) / (rollout_resized.max() - rollout_resized.min() + 1e-8)
    
    return {
        'prediction': pred_class,
        'confidence': float(confidence),
        'raw_prediction': float(prediction),
        'original_image': img_array,
        'gradcam_heatmap': gradcam_heatmap,
        'attention_map': attention_map,
        'fused_map': fused_map,
        'attention_heads': attention_heads,
        'layer_attention': layer_attention,
        'rollout_attention': rollout_resized
    }

def plot_to_base64():
    """Chuyển plot thành base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Đọc và tiền xử lý ảnh
        image = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(image)
        
        # Phân tích toàn diện
        results = analyze_image_comprehensive(img_array)
        
        # Tạo comprehensive visualization
        fig = plt.figure(figsize=(20, 25))
        
        # 1. Kết quả chính và overview
        plt.subplot(5, 4, 1)
        plt.imshow(results['original_image'])
        plt.title(f'Ảnh gốc\nKết quả: {results["prediction"]}\nĐộ tin cậy: {results["confidence"]*100:.1f}%', 
                 fontsize=12, fontweight='bold', color='green' if results['prediction'] == 'Benign' else 'red')
        plt.axis('off')
        
        # 2. Grad-CAM và Attention cơ bản
        plt.subplot(5, 4, 2)
        plt.imshow(results['gradcam_heatmap'], cmap='jet')
        plt.title('Grad-CAM Heatmap\n(Vùng quyết định)', fontsize=12)
        plt.axis('off')
        
        plt.subplot(5, 4, 3)
        plt.imshow(results['attention_map'], cmap='jet')
        plt.title('Attention Map\n(Vùng tập trung)', fontsize=12)
        plt.axis('off')
        
        plt.subplot(5, 4, 4)
        fused_overlay = overlay_alpha_blend(results['original_image'], results['fused_map'], color=(255, 255, 0), alpha=0.6)
        plt.imshow(fused_overlay)
        plt.title('Kết hợp Grad-CAM + Attention\n(Tổng quan)', fontsize=12)
        plt.axis('off')
        
        # 3. Multiple Attention Heads
        for i in range(4):
            plt.subplot(5, 4, 5 + i)
            head_overlay = overlay_alpha_blend(results['original_image'], results['attention_heads'][i], 
                                             color=(0, 255, 0), alpha=0.6)
            plt.imshow(head_overlay)
            plt.title(f'Attention Head {i+1}', fontsize=11)
            plt.axis('off')
        
        # 4. Layer-wise Attention
        layer_names = ['Layer Đầu', 'Layer Giữa 1', 'Layer Giữa 2', 'Layer Cuối']
        for i in range(4):
            plt.subplot(5, 4, 9 + i)
            layer_overlay = overlay_alpha_blend(results['original_image'], results['layer_attention'][i], 
                                              color=(255, 0, 255), alpha=0.6)
            plt.imshow(layer_overlay)
            plt.title(f'{layer_names[i]}', fontsize=11)
            plt.axis('off')
        
        # 5. Overlay comparisons
        plt.subplot(5, 4, 13)
        gradcam_overlay = overlay_alpha_blend(results['original_image'], results['gradcam_heatmap'], 
                                            color=(255, 0, 0), alpha=0.6)
        plt.imshow(gradcam_overlay)
        plt.title('Grad-CAM Overlay\n(Đỏ: Vùng quyết định)', fontsize=11)
        plt.axis('off')
        
        plt.subplot(5, 4, 14)
        attention_overlay = overlay_alpha_blend(results['original_image'], results['attention_map'], 
                                              color=(0, 255, 0), alpha=0.6)
        plt.imshow(attention_overlay)
        plt.title('Attention Overlay\n(Xanh: Vùng tập trung)', fontsize=11)
        plt.axis('off')
        
        # plt.subplot(5, 4, 15)
        # rollout_overlay = overlay_alpha_blend(results['original_image'], results['rollout_attention'], 
        #                                     color=(0, 0, 255), alpha=0.6)
        # plt.imshow(rollout_overlay)
        # plt.title('Attention Rollout\n(Tích hợp tất cả layers)', fontsize=11)
        # plt.axis('off')
        
        # 6. Summary và thông tin
        plt.subplot(5, 4, 16)
        plt.text(0.1, 0.9, 'TÓM TẮT KẾT QUẢ', fontsize=14, fontweight='bold', color='navy')
        plt.text(0.1, 0.7, f'Chẩn đoán: {results["prediction"]}', 
                fontsize=12, fontweight='bold', 
                color='green' if results['prediction'] == 'Benign' else 'red')
        plt.text(0.1, 0.6, f'Độ tin cậy: {results["confidence"]*100:.1f}%', fontsize=12)
        plt.text(0.1, 0.5, f'Giá trị: {results["raw_prediction"]:.4f}', fontsize=11)
        plt.text(0.1, 0.3, 'Giải thích:', fontsize=12, fontweight='bold')
        plt.text(0.1, 0.2, '- Grad-CAM: Vùng ảnh hưởng\n  đến quyết định', fontsize=10)
        plt.text(0.1, 0.1, '- Attention: Vùng model\n  tập trung phân tích', fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        comprehensive_viz_base64 = plot_to_base64()
        
        # Trả về tất cả kết quả
        return jsonify({
            'prediction': results['prediction'],
            'confidence': results['confidence'],
            'raw_prediction': results['raw_prediction'],
            'visualization': f'data:image/png;base64,{comprehensive_viz_base64}',
            'interpretation': {
                'gradcam': 'Phương pháp Grad-CAM chỉ ra các vùng ảnh hưởng trực tiếp đến quyết định phân loại của mô hình',
                'attention': 'Attention maps cho thấy các vùng mà mô hình tập trung phân tích khi xử lý ảnh',
                'heads': 'Mỗi attention head có thể tập trung vào các đặc trưng khác nhau của ảnh',
                'layers': 'Các layer khác nhau xử lý thông tin ở các mức độ trừu tượng khác nhau'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)