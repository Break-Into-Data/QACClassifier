import os
import dotenv
import openai
import bentoml

import tensorflow as tf


dotenv.load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-large"

model = bentoml.tensorflow.load_model("qa_classifier")


def preprocess(text: str):
    vector = client.embeddings.create(
        input=text,
        model=embedding_model,
    ).data[0].embedding
    
    return vector
    
    
def postprocess(prediction):
    return prediction['output_0'].numpy().tolist()[0]
    

@bentoml.service()
class TensorFlowClassifierService:
    model = bentoml.tensorflow.load_model("qa_classifier")
    
    @bentoml.api()
    def classify(self, text):
        processed_text = preprocess(text)
        prediction = self.model.signatures['serving_default'](
            tf.constant([processed_text]))
        result = postprocess(prediction)

        return {"result": result}
