import "./index.css";

const HealthPrawn = () => {
  return (
    <div className="healthy-container">
      <div className="healthyconatiner-1">
        <div>
          <h1 className="health-heading autoshow">Build Strong Muscles:</h1>
          <p className="health-note autoshow">
            Lean protein is excellent for muscle growth and repair,
            <br /> and helps with weight management.
          </p>
        </div>

        <img
          className="girl-image ml-5 autoshow"
          alt="girl-img"
          src="https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739630154/DALL_E_2025-02-15_20.03.08_-_An_animated_girl_wearing_a_research_outfit__including_a_white_lab_coat__safety_goggles__and_gloves__carefully_testing_the_health_condition_of_a_small_-removebg_ztasqj.png"
        />
      </div>
      <div className="healthyconatiner-1">
        <img
          className="girl-image mr-5 autoshow"
          alt="girl-img"
          src="https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739730781/DALL_E_2025-02-16_23.57.53_-_A_colorful_and_realistic_animated_boy_with_big_sparkling_eyes_and_an_excited_expression__eagerly_eating_a_plate_of_prawns._He_has_short__messy_hair_an-removebg-preview_xdg1y9.png"
        />
        <div>
          <h1 className="health-heading ml-5 autoshow">Savor the Flavor:</h1>
          <p className="health-note ml-5 autoshow">
            Enjoy prawns without the calorie worry;
            <br /> they're a flavorful way to get the nutrients you need.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HealthPrawn;
