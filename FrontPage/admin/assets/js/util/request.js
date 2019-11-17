var urls = "http://47.94.235.133:8000/"
function post(url,args,success,fail){
        $.ajax({
            type: 'post',
            url: urls + url,
            data: args,
            async: false,
            timeout:30000, 
            success: function (json){
                success(json)
            },
            error: function (err){
                fail(err)
            }
        })
}