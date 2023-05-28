# Travel Backend README

## 1. 概述

​ 该项目为django+vue3数据大屏，此django后端为前端大屏提供数据支撑，包括但不限于使用api调用规定的数据，调用预测的数据，查询处理的数据

## 2.api介绍

​ 本项目所有的api均可在TravelServer文件夹下的url中看到。

​ 以下为所有api：

```
path("admin/", admin.site.urls),
path("api/data/nmainland/all", api.views.api_nmainland_all),
path('api/data/nmainland/sum/<int:year>', api.views.api_nmainland_sum_year),
path('api/data/nmainland/per/<int:year>', api.views.api_nmainland_per_year),
path('api/data/nmainland/sum/<int:year>/<int:month>', api.views.api_nmainland_sum_month),
path('api/data/nmainland/per/<int:year>/<int:month>', api.views.api_nmainland_per_month),
path("api/data/hotel/all", api.views.api_hotel_all),
path("api/data/hotel/per", api.views.api_hotel_rate),
path("api/data/weather", api.views.api_weather),
path("api/data/country/rate", api.views.api_country_rate),
path('', TemplateView.as_view(template_name='index.html'))
```

### path("admin/", admin.site.urls),

此api打开管理员界面，该站点功能强大，可以直接管理数据库和账户，修改的所有数据可以直接在前端反应出来，并且可以直接修改数据库，对数据库进行增删改查（这不就是老师要的后台管理界面嘛，省得写代码了，多好，还带有权限控制，是不是很棒）

![image-20230408175328237](C:\Users\LIUYAN\AppData\Roaming\Typora\typora-user-images\image-20230408175328237.png)

![image-20230408175258143](C:\Users\LIUYAN\AppData\Roaming\Typora\typora-user-images\image-20230408175258143.png)

![image-20230408175318719](C:\Users\LIUYAN\AppData\Roaming\Typora\typora-user-images\image-20230408175318719.png)

### path('', TemplateView.as_view(template_name='index.html'))

该链接不是api，链接到前端编译好的html入口上，实现前后端一体化，实现django同时启动前后端。

### 其他数据类型api

接下来所有的api均遵循一个原则：api/（返回值类型，data，img，video等）/ 数据库表 / 操作 / 附加条件/附加条件 。。。

例如，本api表示该api寻求返回data类型数据，需要查nmainland表，返回所有数据

其中操作符有 all，per，sum等 ，分别表示查所有，查询同比增长，查询总和

返回值格式如下（所有api均返回json）：

所有all的api返回所有数据库的数据，所以格式和数据库一致

所有per返回{ ‘per’ ：num}，num=原值*100

所有sum返回{ ‘sum’ ：num}，num一般为原值，特殊情况返回原值/10000

**ps：api里的<…>为通配符，格式为<数据类型：变量名称>**

celery启动命令win,linux不需要
celery worker -A tasks --loglevel=info -P eventlet
celery -A tasks worker --loglevel=info -P eventlet

redis
./redis-server.exe redis.windows.conf

## 必须的包
django==3.1.7
dmPython
django_dmPython
eventlet ==latest
celery == latest
pandas
redis
django-pandas
django-cors-headers
numpy(捆绑安装)
scikit_learn