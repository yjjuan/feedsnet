from django.conf.urls import include, url
from django.urls import path

from django.contrib import admin
admin.autodiscover()

import hello.views

# Examples:
# url(r'^$', 'gettingstarted.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', hello.views.index, name='index'),
    url(r'^result/$', hello.views.result, name='result'),
    url(r'^control_1/$', hello.views.control_1, name='control_1'),
    url(r'^control_2/$', hello.views.control_2, name='control_2'),
    path('admin/', admin.site.urls),
]
