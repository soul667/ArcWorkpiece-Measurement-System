在@/antd-demo/src/component/FilterSection.js 中在全屏和裁减中间添加 x_mode y_mode z_mode的下拉选项，同时添加说明。
同时在crop发送的时候
```js
    try {
      console.log('Sending crop request:', JSON.stringify({
        regions
      }));
      const response = await fetch('http://localhost:9304/api/point-cloud/crop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          regions
        })
      });
```
将x_mode y_mode z_mode的信息也发送过去

-----------------------

@/UserInterface/PointCouldProgress.py 中经过测试有如下bug
mode是keep的时候，如果我同时选择了多个区域我的意思是这些区域都要保留。你可能理解错了以为是这些区域取交集，请修改


--------------