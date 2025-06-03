"""
URL configuration for optimuspy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.views import (PasswordChangeView,
                                       PasswordResetCompleteView,
                                       PasswordResetConfirmView,
                                       PasswordResetDoneView,
                                       PasswordResetView,
                                       )
from django.urls import path, include

from web import views
from web.forms import PasswordChangeF, SetPasswordF

# Create your views here.

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('oauth/', include('social_django.urls', namespace='social')),
    path('accounts/login/', views.LogIn.as_view(), name='login'),
    path('accounts/signup/', views.SignUp.as_view(), name='signup'),
    path('accounts/logout/', views.ulogout, name='logout'),
    path('accounts/profile/', views.profile, name='profile'),
    path('accounts/password_change/',
         PasswordChangeView.as_view(
             template_name='pwd/change.html',
             success_url='/tasks/', form_class=PasswordChangeF
         ), name='password_change'),
    path('accounts/password_reset/',
         PasswordResetView.as_view(
             template_name='pwd/reset.html'), name='password_reset'),
    path('accounts/password_reset/done/',
         PasswordResetDoneView.as_view(
             template_name='pwd/reset_done.html'), name='password_reset_done'),
    path('accounts/password_reset/confirm/<uidb64>/<token>/',
         PasswordResetConfirmView.as_view(
             template_name='pwd/reset_confirm.html', form_class=SetPasswordF), name='password_reset_confirm'),
    path('accounts/password_reset/complete/',
         PasswordResetCompleteView.as_view(
             template_name='pwd/reset_complete.html'), name='password_reset_complete'),
    path('tasks/', views.tasks_list, name='list'),
    path('tasks/submit/', views.tasks_submit, name='submit'),
    path('tasks/<str:th>/result', views.tasks_result, name='result'),
    path('tasks/<str:th>/signature/', views.tasks_signature, name='signature'),
    path('tasks/<str:th>/stats/', views.tasks_stats, name='stats'),
    path('download/<int:rid>/', views.result_download, name='download'),
    path('api/auth/', views.api_auth, name='api_auth'),
    path('api/compilers/', views.api_compilers, name='api_compilers'),
    path('api/passes/', views.api_passes, name='api_passes'),
    path('api/tasks/', views.api_tasks, name='api_tasks'),
    path('api/submit/', views.api_submit, name='api_submit'),
    path('api/result/', views.api_result, name='api_result'),
    path('api/download/', views.api_download, name='api_download'),
    path('api/cpuinfo/', views.api_cpuinfo, name='api_cpuinfo')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
