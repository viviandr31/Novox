# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.template import RequestContext
from .forms import DocumentForm
from tb.analyzer3 import analyze3
from datetime import datetime
import nltk
import nltk.data

def get_time_tag():
    return datetime.today().strftime("%Y%m%d_%H%M")


def index(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            analyzed_results = analyze3(request.FILES['docx'])
            # analyze(request.FILES['docx'],
            #         unigram_text=form.cleaned_data['unigram'],
            #         bigram_text=form.cleaned_data['bigram'],
            #         pos_text=form.cleaned_data['pos'],
            #         outpath_count='media/' + count_vector,
            #         outpath_norm='media/' + norm_vector)
            # result = '%s is processed' % request.FILES['docx']
    else:
        form = DocumentForm()  # A empty, unbound form
        result = None
        norm_vector = ''
        count_vector = ''
        analyzed_results = ''

    # Render list page with the documents and the form
    return render_to_response(
        'nlp/index.html',
        locals(),
        context_instance=RequestContext(request)
    )
