import streamlit as st
from openai import OpenAI
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Show title and description.
st.title("🖼 Image captioning")
st.write(
    "Upload an image below, and the AI will caption it for you!"
    )
 
st.write("This image captioning AI uses data from the image's colour values, as well as similarity calculation. Because using something like the Blip2ForConditionalGeneration and AutoProcessor because of a lack of memory allocation possibilities was not possible, the captions may not be very detailed, but they work anyway.")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue. You need a paid subscription to use this webapp.", icon="🗝️")
else:

    # openai_api_key = st.secrets["openai_api_key"]

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # load the clip model
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Let the user upload an image
    uploaded_file = st.file_uploader(
        "Upload an image (.png or .jpg)", type=("png", "jpg")
    )

    
    if uploaded_file is not None:

        # Process the uploaded image and question.

        # Convert the file to an opencv image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(image)

        length = st.radio(
            "Do you want a long or short caption?",
            ["long", "short"]
        )

        if st.button("Generate Caption"):
            try:
                # preprocess image
                inputs = processor(images=image, return_tensors="pt")

                # generated_ids = model.generate(**inputs, max_new_tokens=20)

                # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                # generated_text = generated_text[0].strip()

                # encode
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # possible text labels
                text_labels = [
                    "a photo of an object",
                    "a photo of nature",
                    "a photo of an animal",
                    "a photo of people",
                    "a drawn picture",
                ]

                text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_features = model.get_text_features(**text_inputs)

                # calc similarities between image and text features
                similarity = torch.cosine_similarity(image_features, text_features)
                best_match = text_labels[similarity.argmax().item()]

                question = f"Please write a caption for this image, based on the input data that you got. The content of the image is best described as '{best_match}'. Please only describe the content and not the technical data. You can make the caption rather extensive." if length == "long" else f"Please write a caption for this image, based on the input data that you got, in one sentence. The content of the image is best described as '{best_match}'. Please only describe the content and not the technical data."

                messages = [
                    {
                        "role": "user",
                        "content": f"Here's an image: {img_array} \n\n---\n\n {question}",
                    }
                ]

                # Generate an answer using the OpenAI API.
                stream = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    stream=True,
                )


                # Stream the response to the app using `st.write_stream`.
                st.write_stream(stream)

            except Exception as e:
                st.error(f"Error: {e}")


        # image_decode = uploaded_file.read()

        # messages = [
        #     {
        #         "role": "user",
        #         "content": f"Here's an image: {image_decode} \n\n---\n\n {question}",
        #     }
        # ]

        # # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-4-turbo",
        #     messages=messages,
        #     stream=True,
        # )


        # # Stream the response to the app using `st.write_stream`.
        # st.write_stream(stream)
