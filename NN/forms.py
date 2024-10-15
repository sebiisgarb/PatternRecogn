from django import forms

class ImageForm(forms.Form):
    nr_pers = forms.IntegerField(max_value=40, min_value=1)
    nr_poza = forms.IntegerField(max_value=10, min_value=9)

    