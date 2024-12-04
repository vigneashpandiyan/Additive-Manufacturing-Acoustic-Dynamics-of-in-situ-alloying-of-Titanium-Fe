# Acoustic emission signature of martensitic transformation in Laser Powder Bed Fusion of Ti6Al4V-Fe, supported by operando X-ray diffraction

![Abstract](https://github.com/user-attachments/assets/8bc643ea-10ea-4973-b733-9cacfc157627)

# Journal link

[https://doi.org/10.1016/j.addma.2024.104562](https://doi.org/10.1016/j.addma.2024.104562)

# Overview


This study focuses on investigating Acoustic Emission (AE) monitoring in the Laser Powder Bed Fusion (LPBF) process, using premixed Ti6Al4V-(x wt%) Fe, where x = 0, 3, and 6. By employing a structure-borne AE sensor, we analyze AE data statistically, uncovering notable discrepancies within the 50–750 kHz frequency range. Leveraging Machine Learning (ML) methodologies, we accurately predict composition for particular processing conditions. These fluctuations in AE signals primarily arise from unique microstructural alterations linked to martensitic phase transformation, corroborated by operando synchrotron X-ray diffraction and post-mortem SEM and EBSD analysis. Moreover, cracks are evident at the periphery of the printed parts, stemming from local inadequate heat input during the blending of Ti6Al4V with added Fe powder. These cracks are discerned via AE signals subsequent to the cessation of the laser beam, correlating with the presence of brittle intermetallics at their junction. This study highlights for the first time the potential of AE monitoring in reliably detecting footprints of martensitic transformations during the LPBF process. Additionally, AE is shown to prove valuable for assessing crack formations, particularly in scenarios involving premixed powders and necessitating precise selection of processing parameters, notably at part edges.

![Experimental setup](https://github.com/user-attachments/assets/9ee5e9fa-97bb-4033-b069-ccba22565d0f)


# Experimental procedures

![Byol](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Learning-Coaxial-DED-Process-Zone-Imaging/assets/39007209/12b87183-40e0-43bf-86ed-69e6d2495fe1)

The training of ML algorithms is usually supervised. Given a dataset consisting of an input and corresponding label, under a supervised paradigm, a typical classification algorithm tries to discover the best function that maps the input data to the correct labels. On the contrary, self-supervised learning does not classify the data to its labels. Instead, it learns about functions that map input data to themselves . Self-supervised learning helps reduce the amount of labelling required. Additionally, a model self-supervisedly trained on unlabeled data can be refined on a smaller sample of annotated data. BYOL is a state-of-the-art self-supervised method proposed by researchers in DeepMind and Imperial College that can learn appropriate image representations for many downstream tasks at once and does not require labelled negatives like most contrastive learning methods. The BYOL framework consists of two neural networks, online and target, that interact and learn from each other iteratively through their bootstrap representations, as shown in Figure below. Both networks share the architectures but not the weights. The online network is defined by a set of weights θ and comprises three components: Encoder, projector, and predictor. The architecture of the target network is the same as the online network but with a different set of weights ξ, which are initialized randomly. The online network has an extra Multi-layer Perceptron (MLP) layer, making the two networks asymmetric. During training, an online network is trained from one augmented view of an image to predict a target network representation of the same image under another augmented view. The standard augmentation applied on the actual images is a random crop, jitter, rotation, translation, and others. The objective of the training was to minimize the distance between the embeddings computed from the online and target network. BYOL's popularity stems primarily from its ability to learn representations for various downstream visual computing tasks such as object recognition and semantic segmentation or any other task-specific network, as it gets a better result than training these networks from scratch. As far as this work is concerned, based on the shape of the process zone and the four quality categories that were gathered, BYOL was employed in this work to develop appropriate representations that could be used for in-situ process monitoring.

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

