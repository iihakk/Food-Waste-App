Folder Contents

Report.pdf
A detailed 5-page report covering the system description, GA strategy justification, objectives, data generation, and results interpretation.

Food-waste-reduction.ipynb
A documented Google Colab notebook containing the full implementation, experimental trials, and evolutionary history of the algorithm.

simulation_data.csv
The final generated dataset used for testing, including synthetic store and customer data across Cairo districts.

Wednesday_Presentation.pdf
Presentation slides delivered on Wednesday, December 17.

Contribution_Matrix.url
A link to the Google Sheet detailing work distribution among group members.

Live Links & Dashboard

Deployed Simulator:
https://food-waste-app-optimizers.streamlit.app/

GitHub Repository:
https://github.com/iihakk/Food-Waste-App

Work Distribution Sheet:
Link to Google Sheet

📊 Key Results Summary

The proposed Genetic Algorithm (EMO-Rank) was benchmarked against three alternative strategies. While deterministic models such as Time Decay Urgency minimize physical bag waste, the GA consistently outperforms others in maximizing economic throughput.

Algorithm	Revenue (EGP)	Waste (Bags)	Waste (EGP)	Runtime
Genetic Algorithm	257,071	7,268	337,627	93.170s
Supply Demand Equilibrium	250,259	7,099	338,373	42.073s
Time Decay Urgency	250,034	7,059	337,595	110.394s
Greedy Baseline	226,543	7,237	355,955	16.773s

Data derived from simulation trials conducted on December 17, 2025.

🛠️ Installation and Running Locally

To run the simulation dashboard locally:

Clone the repository

git clone https://github.com/iihakk/Food-Waste-App.git
cd Food-Waste-App


Install dependencies

pip install -r requirements.txt


Launch the Streamlit app

streamlit run app.py

👥 Authors

Salma Elmarakby

Nour Elghaly

Yassin Shamaa

Abdulaziz Al-haidary
