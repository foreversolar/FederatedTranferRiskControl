function login() {
    var phone = $("[name='phone']").val()
    var password = $("[name='password']").val()
    args = {
        "phone": phone,
        "password": password
    }
    post('login', args,
        (data) => {
            data = JSON.parse(data).data
            if(data.state==200){
                localStorage.setItem("userId",data.userId)
                window.location.href = "index.html";
            }
            else alert(data.msg)
        },
        (error) => alert("请求错误")
    )
}