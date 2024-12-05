# Acoustic emission signature of martensitic transformation in Laser Powder Bed Fusion of Ti6Al4V-Fe, supported by operando X-ray diffraction

![image](https://github.com/user-attachments/assets/7dbf50af-a350-43b0-b7ca-3c1d1129b30a)

# Journal link

[https://doi.org/10.1016/j.addma.2024.104562](https://doi.org/10.1016/j.addma.2024.104562)

# Overview

This study focuses on investigating Acoustic Emission (AE) monitoring in the Laser Powder Bed Fusion (LPBF) process. Operando X-ray diffraction was conducted to reveal the microstructure changes associated with phase transformations using premixed Ti6Al4V-(x wt%) Fe, where x = 0, 3, and 6. Taking this as a base-line and by employing a structure-borne AE sensor in off-line experiments, we analyze AE data statistically, uncovering notable discrepancies within the 50–750 kHz frequency range. Leveraging Machine Learning (ML) methodologies, we accurately predict composition for particular processing conditions. These fluctuations in AE signals primarily arise from unique microstructural alterations linked to martensitic phase transformation, corroborated by operando synchrotron X-ray diffraction and post-mortem SEM and EBSD analysis. Moreover, cracks are evident at the periphery of the printed parts, stemming from local inadequate heat input during the blending of Ti6Al4V with added Fe powder. These cracks are discerned via AE signals subsequent to the cessation of the laser beam, correlating with the presence of brittle intermetallics at their junction. This study highlights for the first time the potential of AE monitoring in reliably detecting footprints of martensitic transformations during the LPBF process. Additionally, AE is shown to prove valuable for assessing crack formations, particularly in scenarios involving premixed powders and necessitating precise selection of processing parameters, notably at part edges.

![Experimental setup](https://github.com/user-attachments/assets/9ee5e9fa-97bb-4033-b069-ccba22565d0f)
![Xray](https://github.com/user-attachments/assets/08493967-153b-433f-820e-a8f83a1987a5)

# Experimental procedures

Commercial-grade Ti-6Al-4V ELI (Ti64) powder, sourced from AP&C in Canada, was used as the initial material. The Ti64 powder had a particle size distribution characterized by D90 = 47 µm, D50 = 35 µm, and D10 = 21 µm. To create the pre-mixtures of Ti64–3Fe and Ti64–6Fe, the Ti64 powder was mixed with 3 wt% and 6 wt% of high-purity (99 %) fine Fe particles, respectively. The mixing was carried out in an Ar-sealed tubular mixer for a duration of 2 hr. The Fe powder, sourced from Goodfellow Cambridge Limited, possessed a particle size distribution with D90 = 12.4 µm, D50 = 5.94 µm, and D10 = 2.88 µm. The printing process was performed using a miniaturized LPBF machine under Ar environment with oxygen levels maintained below 0.1 %. Three sets of powder were used for the LPBF process, namely Ti64, Ti64–3Fe, and Ti64–6Fe. For each set, two cuboid geometries (length: 4 mm, and width: 2 mm) were printed with distinct energy densities and their corresponding AE was recorded simultaneously. The energy densities corresponded to CM and KM regimes. The process parameters were chosen to obtain nearly full dense parts and the prints were done on grade 5 Ti64 base plates without any support structures. The selection of the processing parameters was chosen based on the notion of Normalized Enthalpy (NE). A 4 mm unidirectional scanning vector was employed throughout the whole print.The as-built samples were sectioned perpendicular to the laser scanning path along the building direction for microscopy analysis. Electron Backscatter Diffraction (EBSD) characterization was performed using the already described SEM via the Aztec (Oxford Instrument Nanoanalysis) software. EBSD data were taken with a step size of 0.59 µm at 25 kV and 10 nA operation condition. The maps were subsequently processed using an AZtec Crystal (Oxford) plug-in. The parent grains were reconstructed based on the Burgers Orientation Relationships (BORs).

![Microstructure1](https://github.com/user-attachments/assets/a86a46ad-f25a-4c80-8fc3-a45ca7e644db)

It is well-established that solid-state phase transformations generate AE due to the energy release associated with microstructural changes and the related mechanical eventsappropriate representations that could be used for in-situ process monitoring. In displacive transformations, such as martensite formation, the rapid rearrangement of atoms results in a significant energy release, producing intense acoustic signals. Thus, acoustic sensors can detect these waves, offering valuable insights into transformation kinetics by analyzing signal characteristics like amplitude, frequency, and timing. The build plate within the process chamber underwent customization to effectively accommodate the structure-borne AE sensor. Our investigation involves using Convolutional Neural Network (CNN) architectures as automatic feature extractors. This method combines AE data processing with ML algorithms for feature extraction. Contrastive learning is a method employed to train neural networks in discerning positive and negative pairs, with the goal of constructing a feature embedding space where similar instances are grouped closely together while dissimilar ones are positioned farther apart. This approach provides an effective means of automatically deriving feature vectors, which can subsequently be utilized in tandem with ML classifiers. Statistical analysis revealed distinct characteristics for each powder composition and processing parameters. Employing a ML algorithm facilitated the high-accuracy classification of AE signals. Operando X-ray diffraction and post-mortem EBSD characterizations confirmed the effectiveness of in-situ alloying, preventing α’-martensite in KM printing of Ti64–3wt%Fe and Ti64–6wt%Fe. Medium energy density in CM printing was partly effective in Fe mixing, resulting in a mixture of α’ and β phases with varying pro-
portions for each powder composition. Quasi-correlation between operando X-ray diffraction, EBSD, and wavelet analysis demonstrated the existence of high-frequency components up to 380 kHz in samples undergoing martensitic transformation. Examination of raw AE data revealed events occurring when the
laser is switched off, with microscopy confirming crack formation at sample edges. These AE signatures, linked to insufficient heat input and brittle intermetallic formation, underscore the importance of careful processing parameter selection for LPBF contours and complex geometries, especially when using premixed powders.

![Spectrum](https://github.com/user-attachments/assets/861a70d7-d44d-42ab-9b25-fa5ea47da7a4)

# Results

![Crack](https://github.com/user-attachments/assets/4b992c5a-9613-4fb2-a629-65c0719d12d5)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe
cd Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe

python ../Data_preprocessing/Data_prep.py
python ../Feature extraction/Main_features.py
python ../Feature extraction/Main_Features PSD.py
python ../Crack dynamics/Main_Visualize.py
python ../Contrastive Loss/Main.py

```

# Citation
```
@article{esmaeilzadeh2024acoustic,
  title={Acoustic emission signature of martensitic transformation in Laser Powder Bed Fusion of Ti6Al4V-Fe, supported by operando X-ray diffraction},
  author={Esmaeilzadeh, Reza and Pandiyan, Vigneashwara and Van Petegem, Steven and Van der Meer, Mathijs and Nasab, Milad Hamidi and de Formanoir, Charlotte and Jhabvala, Jamasp and Navarre, Claire and Schlenger, Lucas and Richter, Roland and others},
  journal={Additive Manufacturing},
  pages={104562},
  year={2024},
  publisher={Elsevier}
}
```

