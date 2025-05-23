import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./index.css";

const Footer = () => {
  return (
    <div className="footer-section p-5" id="Fotter">
      <Container>
        <Row>
          <Col xs={6} lg={5}>
            <div className="text text-center">
              <img className="footer-image" src="https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739094393/DALL_E_2025-02-08_22.47.59_-_A_modern_and_minimalist_logo_of_a_shrimp_designed_with_sleek_curves_and_clean_lines._The_shrimp_is_stylized_with_a_smooth_flowing_body_a_curled_tai_plppur.webp"/>
              <h1 className="footer-heading font_style">Prawn Disease</h1>
              <p className="footer-sub-heading">Pablo Research</p>
            </div>
          </Col>
          <Col xs={6} lg={4}>
            <h1 className="footer-heading">More From Us</h1>
            <ul className="text-footer">
              <li className="footer_list_text">Our Team</li>
              <li className="footer_list_text">About us</li>
              <li className="footer_list_text">Contact</li>
              <li className="footer_list_text">Give your feedback</li>
            </ul>
          </Col>
          <Col xs={6} lg={3}>
            <ul className="text-footer">
              <li className="footer_list_text">Terms and conditions</li>
              <li className="footer_list_text">Privacy</li>
              <li className="footer_list_text">Accessibility</li>
              <li className="footer_list_text">Advetise with us</li>
              <li className="footer_list_text">Guidance &amp; treatment</li>
              <li className="footer_list_text">From Pablo Research</li>
              <li className="footer_list_text">Blogs</li>
            </ul>
          </Col>
          <Col xs={12}>
            <div>
              <h1 className="footer-heading text-center">Follow Us</h1>
              <div className="d-flex flex-row justify-content-center">
                <div className="footer-icon-container m-4">
                  <i
                    className="fa-brands fa-facebook icon_size"
                    aria-hidden="true"
                  ></i>
                </div>
                <div className="footer-icon-container m-4">
                  <i
                    className="fa-brands fa-whatsapp icon_size"
                    aria-hidden="true"
                  ></i>
                </div>
                <div className="footer-icon-container m-4">
                  <i
                    className="fa-brands fa-instagram icon_size"
                    aria-hidden="true"
                  ></i>
                </div>
                <div className="footer-icon-container m-4">
                  <i
                    className="fa-brands fa-x-twitter icon_size"
                    aria-hidden="true"
                  ></i>
                </div>
              </div>
            </div>
          </Col>
          <Col xs={12}>
            <hr className="hr-line" />
            <div className="fotter-bottom-cntr">
              <p className="footer-copyright-text">
                <i className="fa-regular fa-copyright" aria-hidden="true"></i>
                2025 by Surya, Sponsered DeepSea AI Research. Developed using React.{" "}
              </p>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Footer;
