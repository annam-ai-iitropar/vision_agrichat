import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("best.pt")

class_descriptions = {
    "anthracnose": "Dark, sunken lesions on leaves and stems.",
    "healthy": "This leaf appears healthy with no visible signs of disease.",
    "powdery mildew": "White powdery spots on leaf surfaces.",
    "leaf blight": "Browning or yellowing of leaf margins and tips.",
    "corn rust leaf": "Orange-brown pustules on corn leaves, fungal infection."
}

st.set_page_config(page_title="🌿 CroPulse AI", layout="wide")

st.markdown("<h2 style='text-align: center; color: #228B22;'>🌿 CroPulse AI - The Smart Eye for Crops</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-based disease detection tool for healthier, smarter farming.</p>", unsafe_allow_html=True)

# Sidebar upload
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf or crop image", type=["jpg", "jpeg", "png"])

left_col, right_col = st.columns(2)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with left_col:
        st.image(image, caption="📷 Original Image", use_container_width=True)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with right_col:
        if st.button("🔍 Predict"):
            with st.spinner("Detecting crop diseases..."):
                scale = 1.2
                resized_img = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, resized_img)
                    temp_path = temp_file.name

                try:
                    results = model(temp_path, conf=0.25)
                finally:
                    os.remove(temp_path)

                img_with_boxes = resized_img.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = model.names[cls_id]

                    # Draw box and label
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="✅ Detection Output", use_container_width=True)

                st.markdown("---")
                st.subheader("📋 Detection Summary")
                if results[0].boxes and len(results[0].boxes.cls) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_name = model.names[cls_id]

                        st.write(f"**🧪 Class:** {class_name}")
                        st.write(f"**🔢 Confidence:** {conf * 100:.2f}%")
                        if class_name in class_descriptions:
                            st.info(f"📝 {class_descriptions[class_name]}")
                        else:
                            st.warning("ℹ️ No description available.")
                        st.markdown("---")
                else:
                    st.warning("⚠️ No disease detected in the image.")

st.markdown(
    "<hr><p style='text-align: center; font-size: 13px;'>🚀 Smart Agriculture System | Powered by ANNAM.AI</p>",
    unsafe_allow_html=True
)
