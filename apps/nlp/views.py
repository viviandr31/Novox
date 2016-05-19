# -*- coding: utf-8 -*-
from django.shortcuts import render_to_response
from django.template import RequestContext
from .forms import DocumentForm
from tb.analyzer import analyze
from datetime import datetime


def get_time_tag():
    return datetime.today().strftime("%Y%m%d_%H%M")


def index(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            time_tag = get_time_tag()
            norm_vector = 'norm_vector_%s.csv' % time_tag
            count_vector = 'count_vector_%s.csv' % time_tag
            analyze(request.FILES['docx'],
                    unigram_text=form.cleaned_data['unigram'],
                    bigram_text=form.cleaned_data['bigram'],
                    pos_text=form.cleaned_data['pos'],
                    outpath_count='media/' + count_vector,
                    outpath_norm='media/' + norm_vector)
            result = '%s is processed' % request.FILES['docx']
    else:
        form = DocumentForm()  # A empty, unbound form
        result = None
        norm_vector = ''
        count_vector = ''

    # Render list page with the documents and the form
    return render_to_response(
        'nlp/index.html',
        locals(),
        context_instance=RequestContext(request)
    )
