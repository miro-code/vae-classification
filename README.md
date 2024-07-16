# vae-classification

We use a VAE to encode images and then use the encoding for classification. This setup enables unsupervised learning from unlabelled training samples. 

Run VAE training with to train vae, save it to model_dir and encode the dataset into the latent space
python -m run_vae 

Run Classifier training with
python -m run_classifier 
