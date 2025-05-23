import { Component } from "react";
import "./index.css";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import DiseasesItems from "./DiseasesItems";
import DiseasesCard from "./Diseases-card";

const diseasesList = [
  {
    id: 1,
    name: "White Spot Syndrome (WSS)",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739731676/DALL_E_2025-02-17_00.17.10_-_A_detailed_digital_illustration_of_a_prawn_infected_with_White_Spot_Syndrome_WSS_shown_in_a_wider_view._The_prawn_has_a_translucent_reddish-brown_e_n7twkk.webp",
  },
  {
    id: 2,
    name: "Black Gill Disease",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739731283/DALL_E_2025-02-17_00.10.54_-_A_highly_detailed_close-up_digital_illustration_of_a_prawn_infected_with_Black_Gill_Disease._The_prawn_has_a_reddish-brown_exoskeleton_with_darkened_audt4e.webp",
  },
  {
    id: 3,
    name: "Vibriosis",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739731975/DALL_E_2025-02-17_00.22.40_-_A_detailed_digital_illustration_of_a_prawn_infected_with_Vibriosis._The_prawn_has_darkened_necrotic_patches_on_its_exoskeleton_along_with_red_lesion_hqanhc.webp",
  },
  {
    id: 4,
    name: "Yellow Head Disease (YHD)",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739731909/DALL_E_2025-02-17_00.21.33_-_A_detailed_digital_illustration_of_a_prawn_infected_with_Yellow_Head_Disease_YHD_shown_in_a_slightly_zoomed-out_view._The_prawn_has_a_swollen_yell_lcfvrj.webp",
  },
  {
    id: 5,
    name: "Taura Syndrome",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739732234/DALL_E_2025-02-17_00.26.59_-_A_detailed_digital_illustration_of_a_prawn_infected_with_Taura_Syndrome._The_prawn_has_a_reddish_inflamed_exoskeleton_with_visible_necrotic_patches_a_qqocds.webp",
  },
  {
    id: 6,
    name: "Infectious Hypodermal and Hematopoietic Necrosis (IHHN)",
    diseaseImg:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739732550/DALL_E_2025-02-17_00.32.11_-_A_detailed_digital_illustration_of_a_prawn_infected_with_Infectious_Hypodermal_and_Hematopoietic_Necrosis_IHHN_shown_in_a_slightly_zoomed-out_view_uxnan4.webp",
  },
];

const iniatialisedDiseasesCardList = [
  {
    id: 1,
    heading: "White Spot Disease (WSD)",
    description:
      "White Spot Disease (WSD), caused by the White Spot Syndrome Virus (WSSV), spreads through contaminated water and infected shrimp. It causes white spots, lethargy, and high mortality. Outbreaks are triggered by poor water quality and stress. With no cure, prevention through biosecurity, disease-resistant shrimp, and early detection is key.",
    treatment:
      "There is no effective treatment for White Spot Disease (WSD), so prevention is key. Farmers should implement strict biosecurity, use specific pathogen-free (SPF) shrimp, and disinfect all equipment and water sources. Maintaining stable water quality and avoiding sudden changes helps reduce stress and infection risk. Infected shrimp should be removed immediately, and affected ponds should be drained and disinfected. Probiotics and immune-boosting feed additives can enhance resistance. Natural antiviral agents like seaweed extracts may help prevent infections. Regular health checks and PCR testing can aid early detection and minimize losses.",
    tips: [
      "Use SPF shrimp and disinfect all farm equipment regularly.",
      "Maintain stable water temperature and avoid sudden environmental changes.",
      "Remove infected shrimp immediately and disinfect affected ponds.",
      "Conduct routine health monitoring and use immune-boosting feed.",
    ],
    Symptoms: [
      "White spots (1–2 mm) on the exoskeleton.",
      "Lethargy and weak swimming behavior.",
      "Reduced feeding or complete loss of appetite.",
      "High mortality within 3–10 days of infection.",
      "Soft shell and molting problems.",
      "Reddish body coloration in severe cases.",
      "Shrimp swimming near the water surface.",
      "Sudden population die-offs.",
    ],
    image:
      "https://images.thefishsite.com/fish/articles/Greg-Lutz-column/shrimp-white-spot-virus-credit-D-Lightner.jpeg?width=1300&height=0",
  },
  {
    id: 2,
    heading: "Black Gill Disease",
    description:
      "Black Gill Disease in shrimp causes darkened gills due to infections or poor water quality, leading to respiratory issues, reduced growth, and increased mortality. It can be triggered by bacteria, fungi, protozoa, and stressors like high ammonia or low oxygen. Prevention includes proper pond management, good aeration, and water quality control. Early detection and corrective actions, such as using probiotics, can help reduce its impact.",
    treatment:
      "Black Gill Disease management requires addressing the root cause, such as bacterial, fungal, or parasitic infections. Improving water quality by monitoring ammonia, nitrite, and oxygen levels is crucial. Regular water exchange, proper aeration, and pH balance help prevent stress. Probiotics and antibacterial treatments can reduce bacterial infections, while antifungal or antiparasitic treatments may be needed for fungal or protozoan causes. Proper stocking densities, sediment removal, and pond cleaning minimize stress and pathogen buildup. Strengthening shrimp immunity with quality feed and routine health checks can prevent outbreaks. Biosecurity measures are essential to reduce risks.",
    tips: [
      "Maintain optimal water quality by monitoring ammonia and oxygen levels.",
      "Use probiotics and immune-boosting feed to strengthen shrimp resistance.",
      "Remove infected shrimp immediately to prevent the spread of disease.",
      "Avoid overstocking and ensure proper aeration in shrimp ponds.",
    ],
    Symptoms: [
      "Dark or blackened gills.",
      "Reduced oxygen intake.",
      "Lethargy and weak swimming behavior.",
      "Slow or reduced feeding.",
      "Shell discoloration in severe cases.",
      "Gills clogged with organic matter.",
      "Increased vulnerability to secondary infections.",
      "High mortality in extreme cases.",
    ],
    image:
      "https://www.researchgate.net/profile/Jill-Gambill/publication/303013758/figure/fig1/AS:361155353694208@1463117554943/A-Georgia-white-shrimp-displays-symptoms-of-the-Black-Gill-condition-around-its-gills.png",
  },
  {
    id: 3,
    heading: "Vibriosis",
    description:
      "Vibriosis, caused by *Vibrio* species (e.g., *V. harveyi*, *V. parahaemolyticus*, *V. vulnificus*), occurs when shrimp are stressed by poor water quality, overcrowding, or environmental changes. It leads to symptoms like shell disease, tail rot, septicemia, lethargy, and high mortality. In severe cases, internal organ failure causes die-offs. The disease spreads through contaminated water, infected shrimp, and biofilm in equipment. Preventative measures include maintaining optimal water quality and strengthening shrimp immunity.",
    treatment:
      "Vibriosis can be managed by prioritizing natural treatments and preventive measures over antibiotics to prevent resistance. Probiotics help control Vibrio populations, while maintaining water quality by monitoring ammonia, nitrite, and pH levels reduces stress. Proper aeration, water exchange, and biosecurity (e.g., disinfecting equipment) are essential. Removing dead or infected shrimp prevents further spread. Natural antimicrobials like garlic extract and turmeric boost immunity, while immunostimulants such as beta-glucans and vitamin C enhance resistance. A balanced diet, avoiding overfeeding, and proper sediment and biofilm management help prevent Vibrio outbreaks.",
    tips: [
      "Use probiotics to control Vibrio populations and improve shrimp health.",
      "Maintain good water quality and avoid overfeeding.",
      "Remove dead shrimp immediately to prevent bacterial spread.",
      "Strengthen shrimp immunity with vitamins, garlic extract, and natural antimicrobials.",
    ],
    Symptoms: [
      "Lethargy and reduced activity.",
      "Loss of appetite and weak feeding behavior.",
      "Reddening of the body and tail.",
      "Soft shell and poor molting.",
      "Lesions or ulcers on the shell.",
      "Swollen hepatopancreas (digestive organ).",
      "Cloudy or darkened gills.",
      "High mortality if untreated.",
    ],
    image:
      "https://static.wixstatic.com/media/2a5991_7afb512745804c9dab2941f8bf9d00d5~mv2.jpg/v1/fill/w_539,h_490,al_c,lg_1,q_80,enc_auto/2a5991_7afb512745804c9dab2941f8bf9d00d5~mv2.jpg",
  },
  {
    id: 4,
    heading: "Yellow Head Disease (YHD)",
    description:
      "Yellow Head Disease (YHD) is a viral infection caused by the *Yellow Head Virus* (YHV) that affects shrimp species like *Penaeus monodon* and *Penaeus vannamei*. The disease targets the hepatopancreas and gills, causing yellow discoloration, erratic swimming, loss of appetite, and swelling. YHD spreads rapidly through infected shrimp, contaminated water, and carrier species like crabs. Stress, overcrowding, and poor management increase outbreak risks. Early detection and biosecurity measures are essential to prevent mass mortality, which can occur within 3 to 5 days.",
    treatment:
      "There is no cure for Yellow Head Disease (YHD), so prevention is key. Farmers should enforce strict biosecurity measures, such as using pathogen-free shrimp, disinfecting equipment, and managing water quality by maintaining optimal pH, temperature, salinity, and oxygen levels. Sudden environmental changes should be avoided. Infected shrimp should be removed, and ponds should be disinfected before restocking. Probiotics, immune-boosting feed, and selective breeding for YHV-resistant shrimp can help reduce infection risks. Regular health monitoring and PCR testing are essential for early detection. Strengthening shrimp immunity with vitamins, minerals, and herbal extracts can further aid in prevention.",
    tips: [
      "Use SPF shrimp and maintain strict biosecurity measures.",
      "Keep water quality stable and avoid sudden environmental changes.",
      "Remove infected shrimp immediately and disinfect affected ponds.",
      "Enhance shrimp immunity with high-quality feed and probiotics.",
    ],
    Symptoms: [
      "Yellowish or pale discoloration of the head.",
      "Swelling of the cephalothorax (head region).",
      "Reduced feeding and sluggish behavior.",
      "Soft shell and molting problems.",
      "Rapid mortality, sometimes within 24 hours.",
      "Shrimp floating near the water surface.",
      "Mass die-offs in ponds.",
      "Internal organs appear pale and degenerated.",
    ],
    image:
      "https://res.cloudinary.com/dpvrtbqxt/image/upload/v1739783315/4-Figure1-1_3_pah82d.png",
  },
  {
    id: 5,
    heading: "Taura Syndrome (TSV)",
    description:
      "Taura Syndrome Virus (TSV) is a viral disease affecting *Penaeus vannamei*, spreading through contaminated water, infected shrimp, and predators like birds and crabs. Environmental stressors like poor water quality and overcrowding increase outbreak risks. The disease has three phases: 1. **Acute:** Soft shells, reddish discoloration, erratic swimming, and high mortality. 2. **Transition:** Some recovery, but dark melanized spots appear. 3. **Chronic:** Shrimp seem normal but are virus carriers. TSV causes significant economic losses in shrimp farming, and prevention through biosecurity and resistant shrimp strains is essential.",
    treatment:
      "There is no specific treatment for Taura Syndrome, so prevention is key. Strict biosecurity is crucial, including sourcing shrimp from TSV-free hatcheries and avoiding wild crustaceans as carriers. Maintaining optimal water quality by monitoring ammonia, nitrite, pH, and aeration helps reduce stress and infection risks. Consistent water temperature and the use of TSV-resistant shrimp strains also aid prevention. Probiotics and immune-boosting feed improve shrimp immunity, while removing infected shrimp and disinfecting equipment prevent further spread. Quarantining new shrimp ensures disease-free stock before adding them to main ponds.",
    tips: [
      "Use only certified TSV-free shrimp stock.",
      "Maintain stable water quality and avoid sudden temperature fluctuations.",
      "Remove infected shrimp immediately to reduce the risk of further spread.",
      "Implement strict biosecurity and regular pond disinfection practices.",
    ],
    Symptoms: [
      "Reddish coloration on the body and tail.",
      "Soft shell and weak exoskeleton.",
      "Lethargy and reduced movement.",
      "Lethargy and reduced movement.",
      "Blackened gills in later stages.",
      "Sudden high mortality in juvenile shrimp.",
      "Molting issues leading to deformities.",
      "Small melanized (black) spots on the shell.",
    ],
    image: "https://library.enaca.org/Health/FieldGuide/images/tau2.jpg",
  },
  {
    id: 6,
    heading: "Infectious Hypodermal and Hematopoietic Necrosis (IHHN)",
    description:
      "Infectious Hypodermal and Hematopoietic Necrosis (IHHN) is a viral disease that affects shrimp, especially *Penaeus vannamei* and *Penaeus stylirostris*. Caused by the IHHN virus, it leads to deformities, reduced growth, and high mortality, particularly in juvenile shrimp. The disease causes runt deformity syndrome (RDS), with bent rostrums and uneven growth. IHHN spreads through infected shrimp, contaminated water, and carrier species like crabs. There is no cure, so prevention through strict biosecurity measures and using virus-free shrimp stocks is essential to control outbreaks.",
    treatment:
      "There is no direct treatment for IHHN, so prevention is key. Farmers should use specific pathogen-free (SPF) shrimp and quarantine new stock before pond introduction. Maintaining optimal water quality through regular monitoring of pH, dissolved oxygen, and ammonia levels helps reduce stress and prevent infections. Proper pond management, including water exchange and sediment removal, also minimizes viral risks. Infected shrimp should be removed and disposed of, and all equipment should be disinfected. Probiotics and immune-boosting feed can enhance shrimp immunity, and selective breeding of IHHN-tolerant strains can provide long-term solutions. Regular disease screening is essential for early detection. ",
    tips: [
      " Use only specific pathogen-free (SPF) shrimp stock. ",
      "Maintain optimal water quality and stable pond conditions.",
      "Regularly disinfect equipment and remove infected shrimp.",
      "Avoid co-culturing shrimp with wild crustaceans to reduce transmission risks.",
    ],
    Symptoms: [
      "Deformities in the shell and rostrum.",
      "Slow growth and stunted development.",
      "Soft shell and weak exoskeleton.",
      "Reduced activity and lethargy.",
      "Molting problems.",
      "High mortality in juvenile shrimp.",
      "Blackened or darkened tissues.",
      "Surviving shrimp may carry the virus without symptoms.",
    ],
    image:
      "https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-1-0716-3032-7_9/MediaObjects/524474_1_En_9_Fig1_HTML.jpg",
  },
];

class Disease extends Component {
  state = { isClicked: false, diseasesCardList: "" };

  viewDetails = (id) => {
    const filteredData = iniatialisedDiseasesCardList.filter(
      (each) => each.id === id
    );
    this.setState({ isClicked: true, diseasesCardList: filteredData });
  };

  backToDiseasesPage = () => {
    this.setState({ isClicked: false });
  };

  render() {
    const { isClicked, diseasesCardList } = this.state;
    return (
      <div className="disease-container" id="Disease">
        {isClicked ? (
          <DiseasesCard
            key={diseasesCardList.id}
            cardDetails={diseasesCardList[0]}
            backToDiseasesPage={this.backToDiseasesPage}
          />
        ) : (
          <div>
            <Container>
              <Row>
                <Col>
                  <h1 className="Disease-heading autoshow">Diseases</h1>
                </Col>
              </Row>
              <Row>
                <ul className="card-container">
                  {diseasesList.map((each) => (
                    <DiseasesItems
                      key={each.id}
                      diseaseList={each}
                      viewDetails={this.viewDetails}
                    />
                  ))}
                </ul>
              </Row>
            </Container>
          </div>
        )}
      </div>
    );
  }
}

export default Disease;
