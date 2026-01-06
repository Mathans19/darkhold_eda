from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('clear-session/', views.clear_session, name='clear_session'),
    path('report/', views.report, name='report'),
    path('clean/', views.clean_data, name='clean'),
    path('visualize/', views.visualize, name='visualize'),
    path('ai-insights/', views.ai_insights, name='ai_insights'),
]
# Force Reload

