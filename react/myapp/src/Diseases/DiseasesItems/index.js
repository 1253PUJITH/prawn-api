import "./index.css";

const DiseasesItems = (props) => {
  const { diseaseList, viewDetails } = props;
  const { id, diseaseImg, name } = diseaseList;
  const onclickButton = () => {
    viewDetails(id);
  };

  return (
    <li className="disease-card col-lg-3 autoshow ">
      <img className="card-image" src={diseaseImg} alt="disease-image" />
      <p className="disease-name">{name}</p>
      <button className="button" onClick={onclickButton}>
        View <i className="arrow fa-solid fa-arrow-right"></i>
      </button>
    </li>
  );
};

export default DiseasesItems;
