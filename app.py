import streamlit as st
import os
import pandas as pd
from PIL import Image
import pickle
from src import Config, WasteClassifier, Visualizer

class StreamlitApp:
    """Streamlit web application"""
    
    def __init__(self):
        self.config = Config()
        self.classifier = WasteClassifier(self.config)
        self.visualizer = Visualizer(self.config)
        
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Waste Classification System",
            page_icon="♻️",
            layout="wide"
        )
        
        # Header
        st.title("🗂️ Smart Waste Classification System")
        st.markdown("""
        This AI-powered system classifies waste into different categories to help with recycling and waste management.
        Upload an image of waste material and get instant classification results!
        """)
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", 
                                   ["Classification", "About", "Statistics"])
        
        if page == "Classification":
            self.classification_page()
        elif page == "About":
            self.about_page()
        elif page == "Statistics":
            self.statistics_page()
    
    def classification_page(self):
        """Main classification page"""
        st.header("📸 Upload and Classify Waste")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of waste material for classification"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("Classification Results")
                
                # Make prediction
                with st.spinner("Classifying..."):
                    try:
                        predicted_class, confidence, all_predictions = self.classifier.predict(image)
                        
                        # Display results
                        st.success(f"**Predicted Class:** {predicted_class.title()}")
                        st.info(f"**Confidence:** {confidence:.2%}")

                        self.show_recycling_info(predicted_class)
                        
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
    
    def show_recycling_info(self, waste_type):
        """Show recycling information based on waste type"""
        recycling_info = {
            'cardboard': {
                'icon': '📦',
                'description': 'Cardboard is highly recyclable!',
                'tips': ['Remove any tape or labels', 'Flatten boxes to save space', 'Keep dry and clean']
            },
            'glass': {
                'icon': '🍾',
                'description': 'Glass can be recycled indefinitely!',
                'tips': ['Remove caps and lids', 'Rinse containers', 'Separate by color if required']
            },
            'metal': {
                'icon': '🥫',
                'description': 'Metal is one of the most recyclable materials!',
                'tips': ['Clean containers thoroughly', 'Remove labels if possible', 'Separate different metal types']
            },
            'paper': {
                'icon': '📄',
                'description': 'Paper recycling saves trees and energy!',
                'tips': ['Remove staples and clips', 'Keep paper dry', 'Separate different paper types']
            },
            'plastic': {
                'icon': '🥤',
                'description': 'Check the recycling number on plastic items!',
                'tips': ['Clean containers', 'Remove caps and lids', 'Check local recycling guidelines']
            },
            'trash': {
                'icon': '🗑️',
                'description': 'This item is not typically recyclable.',
                'tips': ['Consider if it can be reused', 'Look for special disposal programs', 'Minimize future waste']
            }
        }
        
        info = recycling_info.get(waste_type, recycling_info['trash'])
        
        st.markdown("---")
        st.subheader(f"{info['icon']} Recycling Information")
        st.write(info['description'])
        
        st.markdown("**Tips:**")
        for tip in info['tips']:
            st.markdown(f"• {tip}")
    
    def about_page(self):
        """About page"""
        st.header("📋 About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This waste classification system uses deep learning to automatically categorize waste materials 
        into different types, helping individuals and organizations make better recycling decisions.
        """)
    
    def statistics_page(self):
        """Statistics and analytics page"""
        st.header("📈 Model Performance & Statistics")
        
        # Check if training history exists
        if os.path.exists(self.config.HISTORY_PATH):
            with open(self.config.HISTORY_PATH, 'rb') as f:
                history = pickle.load(f)
            
            # Plot training history
            fig = self.visualizer.plot_training_history(history)
            st.pyplot(fig)
        else:
            st.info("No training history found. Train the model first to see statistics.")

def main():
    """Main function to run the application"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()