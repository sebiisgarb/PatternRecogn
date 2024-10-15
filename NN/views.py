import os
from django.shortcuts import render
from .forms import ImageForm
from django.views import View
from PIL import Image


class ImageView(View):

    image_path = ''
    def get(self, request):
        form = ImageForm()
        return render(request, 'NN/nn.html', {
            'form': form,
            'valid': False
        })
    
    def post(self, request):
        form = ImageForm(request.POST)
        if form.is_valid():

            nr_pers = form.cleaned_data['nr_pers']
            nr_poza = form.cleaned_data['nr_poza']

            pgm_file_path = os.path.join('static/att_faces', f's{nr_pers}', f'{nr_poza}.pgm')
            img = Image.open(pgm_file_path)
            png_image_path = os.path.join('static/att_faces', f's{nr_pers}', f'{nr_poza}.png')
            img.save(png_image_path)

            image_url = f's{nr_pers}/{nr_poza}.png'
            
            return render(request, 'NN/nn.html', {
                'form': form,
                'valid': True,
                'image_path': image_url
            })
        else:
            return render(request, 'NN/nn.html', {
            'form': form,
            'valid': False
        })