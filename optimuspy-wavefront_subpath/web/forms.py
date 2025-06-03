
from crispy_forms.bootstrap import InlineCheckboxes
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Div, Fieldset, Layout, Submit
from django import forms
from django.conf import settings
from django.contrib.auth.forms import (PasswordChangeForm, SetPasswordForm,
                                       UserCreationForm)

from web.models import Task, User
from web.ops.compilers import SubmitFormCflags
from web.ops.passes import Passes


class MultipleFileInput(forms.ClearableFileInput):
    """https://github.com/django/django/commit/eed53d0011622e70b936e203005f0e6f4ac48965#"""
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class SignUpForm(UserCreationForm):
    email = forms.EmailField(label='Адрес электронной почты', required=True)

    def __init__(self, *args, **kwargs):
        super(UserCreationForm, self).__init__(*args, **kwargs)
        for name in ('username', 'email', 'password1', 'password2'):
            self.fields[name].help_text = None

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


class SubmitForm(forms.Form):
    title = forms.CharField(max_length=80, label='Имя загрузки (пустое будет заменено хешем)', required=False)
    tests = forms.IntegerField(min_value=1, max_value=100, initial=10, label='Число запусков программы для усреднения времени')
    files = MultipleFileField(label='Файлы')

    compilers = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[
        (str(i.value), i.name) for i in settings.COMPILERS
    ], label='')

    cflags = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[
        (i.name, i.name) for i in SubmitFormCflags
    ], label='')

    passes = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, choices=[
        (str(i.value), i.desc) for i in settings.OPS_PASSES
    ], label='')

    worker_types = forms.MultipleChoiceField(
        choices=Task.WORKER_TYPE,
        label='',
        initial=['default'],
        widget=forms.CheckboxSelectMultiple
    )

    def __init__(self, u: User, *args, **kwargs):
        super().__init__(*args, **kwargs)
        task = Task.objects.filter(user=u).last()
        if task:
            self.fields['compilers'].initial = [str(i) for i in task.compilers]
            self.fields['cflags'].initial = task.cflags
            self.fields['passes'].initial = [str(Passes(i).value) for i in task.passes]
            self.fields['worker_types'].initial = task.worker_types
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'title',
            'tests',
            'files',
            Fieldset(
                'Компиляторы',
                InlineCheckboxes(
                    'compilers'
                ),
            ),
            Fieldset(
                'Уровни оптимизации',
                InlineCheckboxes(
                    'cflags'
                ),
            ),
            Fieldset(
                'Проходы',
                InlineCheckboxes(
                    'passes'
                ),
            ),
            Fieldset(
                'Типы рабочих узлов',
                InlineCheckboxes(
                    'worker_types'
                ),
            ),
            Submit('submit', 'Отправить', css_class='btn btn-dark btn-lg mt-3'),
        )


class SignatureChoiceForm(forms.Form):
    def __init__(self, choices, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.fields['choice'] = forms.CharField(label='', widget=forms.Select(choices=choices), required=True)


class PasswordChangeF(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super(PasswordChangeForm, self).__init__(*args, **kwargs)
        for name in ('old_password', 'new_password1', 'new_password2'):
            self.fields[name].help_text = None


class SetPasswordF(SetPasswordForm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for name in ('new_password1', 'new_password2'):
            self.fields[name].help_text = None


class ThemeForm(forms.Form):
    choice = forms.CharField(label='Тема', widget=forms.Select(choices=[
        ('dark', 'Темная'),
        ('light', 'Светлая')
    ]), required=False)

    def __init__(self, cookie: dict[str, str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['choice'].initial = cookie['theme']
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Div(
            'choice',
            Submit('submit', 'Применить', css_class='btn btn-dark btn-m mt-3'), css_class='d-flex align-items-center justify-content-center')
        )
