import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./index.css";
const DiseasesCard = (props) => {
  const { backToDiseasesPage, cardDetails } = props;
  const { heading, image, description, treatment, tips, Symptoms } =
    cardDetails;
  const onclickButton = () => {
    backToDiseasesPage();
  };
  return (
    <div className="disease-card-conatiner">
      <Container>
        <Row>
          <Col lg={12}>
            <h1 className="main-heading">{heading}</h1>
          </Col>
          <Col lg={4}>
            <div className="left-conatiner">
              <img className="video" alt={heading} src={image} />
              <p className="heading">Symptoms:</p>
              <ul className="symtom-list">
                {Symptoms.map((symptom, index) => (
                  <li key={index}>{symptom}</li>
                ))}
              </ul>
            </div>
          </Col>
          <Col lg={8}>
            <div className="right-container">
              <h1 className="heading">Description</h1>
              <p className="decription">{description}</p>
              <h1 className="heading">Treatment </h1>
              <p className="decription">{treatment}</p>
              <h1 className="heading">Tips & Suggestions:</h1>
              <ul className="decription">
                {tips.map((tip, index) => (
                  <li key={index}>{tip}</li>
                ))}
              </ul>
            </div>
          </Col>
          <Col lg={12}>
            <div className="btn-container">
              <button
                className="back-button"
                type="button"
                onClick={onclickButton}
              >
                Back
              </button>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default DiseasesCard;
