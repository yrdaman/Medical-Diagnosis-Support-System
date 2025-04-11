from tkinter.font import names

from django.urls import path
from .views import index, analyze_symptoms, speech_to_text, results, submit_feedback, signup_view, login_view, logout_view

urlpatterns = [
    path('', index, name='index'),  # Fixed function call
    path('analyze/', analyze_symptoms, name='analyze_symptoms'),
    path("speech-to-text/", speech_to_text, name="speech_to_text"),
    path("results/", results, name="results"),
    path("submit_feedback/", submit_feedback, name="submit_feedback"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
]
