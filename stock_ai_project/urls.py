from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('stock/', include('stock_data.urls')),  # 包含 stock_data 应用的路由
]
