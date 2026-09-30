"""Optional trained-model inference modules.

``model.py`` holds the small SRCNN loader used by ``trained_srcnn``;
``opensr.py`` wraps the optional ESA OpenSR LDSR-S2 latent-diffusion model.
Neither module is imported at application startup: the heavy dependencies are
imported lazily inside the functions that need them.
"""
