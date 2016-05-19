# -*- coding: utf-8 -*-
from django import forms


class DocumentForm(forms.Form):
    docx = forms.FileField(
        label='Select a docx file to be analyzed',
    )
    pos = forms.CharField(label='POS', widget=forms.Textarea,
                          initial='cc cd dt ex fw in jj jjr jjs ls md nn nns '
                                  'nnp nnps pdt pos prp prp$ rb rbr rbs rp '
                                  'sym to uh vb vbd vbg vbn vbp vbz wdt wp '
                                  'wp$ wrb'
                          )
    bigram = forms.CharField(label='BIGRAM', widget=forms.Textarea,
                             initial='for example\n'
                             'for instance\n'
                             'such as'
                             )
    unigram = forms.CharField(label='UNIGRAM', widget=forms.Textarea,
                              initial='but and if or when this the an a that '
                              'at to on of by with for in from there not out '
                              'as more no his you him she they their her i '
                              'he who we which it one all '
                              'however . ? ! : ; , - () [] {} < > " \' /'
                              )
