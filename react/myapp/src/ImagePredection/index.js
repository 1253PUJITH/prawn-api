import React, { useState } from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./index.css";
import axios from "axios";
import diseasesData from "./diseasesData"; // Import the diseases data

const ImagePredection = () => {
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [probabilities, setProbabilities] = useState({});
  const [bestModel, setBestModel] = useState("");
  const [diseaseInfo, setDiseaseInfo] = useState({ description: "", tips: [] });
  const [labeledImage, setLabeledImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // 👈 Loading state

  const onAddImage = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const imageUrl = URL.createObjectURL(file);
      setImage(imageUrl);
      setMessage("Image uploaded successfully! ✅");
    }
  };

  const onPredict = async () => {
    if (!selectedFile) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setIsLoading(true); // 👈 Show loading before API call

      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setTimeout(() => {
        const { predicted_disease, probabilities, labeled_image_base64, best_model } = response.data;

        setPrediction(predicted_disease);
        setProbabilities(probabilities);
        setLabeledImage(labeled_image_base64);
        setBestModel(best_model);

        const diseaseDetails = diseasesData[predicted_disease] || { description: "No description available.", tips: [] };
        setDiseaseInfo(diseaseDetails);

        setIsLoading(false); // 👈 Stop loading after delay
      }, 5000); // 👈 10 seconds delay
    } catch (error) {
      console.error("Error during prediction", error);
      alert("There was an error with the prediction. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <div className="predection-container" id="Prediction">
      <Container>
        <Row>
          <Col lg={12}>
            <h1 className="predict-heading">Prediction</h1>
          </Col>
          <Col lg={5}>
            <div className="image-predection-container">
              <div className="input-container">
                <input
                  className="input"
                  id="upload-button"
                  type="file"
                  accept="image/*"
                  onChange={onAddImage}
                />
              </div>
              <div className="image-container">
                {image && <img src={image} alt="Uploaded" className="uploaded-image" />}
              </div>
              <div className="message-container">
                {message && <p className="success-message">{message}</p>}
              </div>

              <div>
                <div className="btn-container">
                  <button className="predict-btn Ml" onClick={onPredict}>
                    Predict
                  </button>
                </div>
                <p className="note-2">
                  Note:
                  <br />
                  For the best prediction and accurate results, please upload clear images of the prawn without any distractions. Ensure good lighting and focus on the affected area to improve probability scores and overall detection quality.
                </p>
                <p className="note">
                  Currently, this system detects Prawn Health Conditions, classifying them as Healthy, Black Gill, Black Spot , White Spot Disease.
                  Simply upload an image to get instant results.
                </p>
              </div>
            </div>
          </Col>
          <Col lg={7}>
            <div className="output-container">
              <Row>
                <Col lg={12}>
                  <div className="output-heading">
                    <h1>{isLoading ? "Loading..." : prediction || "Output"}</h1>
                  </div>
                </Col>
                <Col lg={6}>
                  <div className="output-predection">
                    <h1 className="result-heading">Probability:</h1>
                    <div className="result-container">
                      {/* Empty content initially */}
                      {isLoading ? (
                        <p> </p>
                      ) : (
                        <>
                          {bestModel && <p><strong>Model:</strong> {bestModel}</p>}
                          {Object.keys(probabilities).length > 0 && (
                            <>
                              {Object.entries(probabilities)
                                .sort((a, b) => b[1] - a[1])
                                .map(([disease, prob]) => (
                                  <p key={disease}>
                                    {disease}: {prob.toFixed(4)}
                                  </p>
                                ))}
                            </>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </Col>
                <Col lg={6}>
                  <div className="labeled_images">
                    <h1 className="result-heading">Labeled Image:</h1>
                    <div className="resultlabel-container">
                      {/* Empty content initially */}
                      {isLoading ? (
                        <p> </p>
                      ) : (
                        <div className="result-label">
                          {labeledImage && <img src={`data:image/png;base64,${labeledImage}`} alt="Labeled Disease Area" className="labeled-image" />}
                        </div>
                      )}
                    </div>
                  </div>
                </Col>
                <Col lg={12}>
                  <div className="output-decription">
                    <h1 className="result-heading">Description:</h1>
                    <div className="result-container">
                      {/* Empty content initially */}
                      {isLoading ? (
                        <p> </p>
                      ) : (
                        <>
                          <p className="description-text">{diseaseInfo.description}</p>
                          {diseaseInfo.tips && diseaseInfo.tips.length > 0 && (
                            <div className="prevention-tips">
                              <h5><strong>Prevention Tips:</strong></h5>
                              <ul>
                                {diseaseInfo.tips.map((tip, index) => (
                                  <li key={index}>{tip}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </Col>
              </Row>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default ImagePredection;
