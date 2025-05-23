import { Component } from "react";
import "./index.css";
import React from "react";
import { Navbar, Nav, Container } from "react-bootstrap";

class Home extends Component {
  render() {
    return (
      <div className="bg-container" id="Home">
        <Navbar id="navelement" expand="lg">
          <Container>
            <Navbar.Brand href="#home">
              <img
                className="logo"
                src="https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739094393/DALL_E_2025-02-08_22.47.59_-_A_modern_and_minimalist_logo_of_a_shrimp_designed_with_sleek_curves_and_clean_lines._The_shrimp_is_stylized_with_a_smooth_flowing_body_a_curled_tai_plppur.webp"
              />
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="m-auto">
                <Nav.Link id="text" href="#Home">
                  Home
                </Nav.Link>
                <Nav.Link id="text" href="#Prediction">
                  Prediction
                </Nav.Link>
                <Nav.Link id="text" href="#Disease">
                  Disease
                </Nav.Link>
                <Nav.Link id="text" href="#Fotter">
                  contact
                </Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>
        <div className="home-background">
          <div className="text-container">
            <p className="top-text">Sustainable Aquaculture Starts Here</p>
            <h1 className="home-heading">
              Detect Early,
              <br /> Protect More
            </h1>
            <p className="bottom-text">Prawn Disease Monitoring</p>
          </div>
        </div>
      </div>
    );
  }
}

export default Home;
