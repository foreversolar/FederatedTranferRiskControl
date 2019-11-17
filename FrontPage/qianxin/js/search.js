function team() {
    $("#team").addClass("active")
    $("#product").removeClass("active")
    $("#company").removeClass("active")

    $("#teamP").css("display", "block")
    $("#productP").css("display", "none")
    $("#companyP").css("display", "none")
}
function product() {
    $("#team").removeClass("active")
    $("#product").addClass("active")
    $("#company").removeClass("active")

    $("#teamP").css("display", "none")
    $("#productP").css("display", "block")
    $("#companyP").css("display", "none")
}
function company() {
    $("#team").removeClass("active")
    $("#product").removeClass("active")
    $("#company").addClass("active")

    $("#teamP").css("display", "none")
    $("#productP").css("display", "none")
    $("#companyP").css("display", "block")
}
function Loading(){
    $("#loading").css("display", "block")
    $('#cover').css('display','block'); //显示遮罩层
    $('#cover').css('height',document.body.clientHeight+100+'px'); //设置遮罩层的高度为当前页面高度
    $('#header').css('opacity',"0.3"); 
}
function unLoading(){
    $("#loading").css("display", "none")
    $('#cover').css('display','none'); //显示遮罩层
    $('#cover').css('height',0+'px'); //设置遮罩层的高度为当前页面高度
    $('#header').css('opacity',"1"); 
}
function addWorker() {
    var worker = "<label><span>员工姓名:</span><input type='text' name='员工姓名' placeholder='请输入员工姓名' required></label>"
    var workerId = "<label> <span>员工身份证号:</span><input type='text' name='员工身份证号' placeholder='请输入员工身份证号' required></label>"
    var count = "<label> <span>学术交流次数:</span><input type='text' name='学术交流次数' placeholder='请输入员工学术交流次数' required></label>"
    
    var count1 = "<label> <span>商业旅行时间:</span><input type='text' name='商业旅行时间' placeholder='请输入员工商业旅行时间' required></label>"
    var count2 = "<label> <span>月收入:</span><input type='text' name='月收入' placeholder='请输入员工月收入' required></label>"
    var count3 = "<label> <span>任职过多少公司:</span><input type='text' name='任职过多少公司' placeholder='请输入员工任职过多少公司' required></label>"
    var count4 = "<label> <span>在职时常:</span><input type='text' name='在职时常' placeholder='请输入员工在职时常' required></label>"

    var wokerTime = "<label style='margin-bottom:40px'><span>培训时常:</span><input type='number' name='培训时常' placeholder='培训时常' min='0' required></label>"
    var workers = $("#worker");
    if (workers.children().length < 40) {
        workers.append(worker, workerId, count,count1,count2,count3,count4, wokerTime);
    }
    else {
        alert("最多只能添加5个员工")
    }
}
function subm() {
    team()                              //切换图表位置
    Loading()                           //加载
    //获得产品信息
    product = ""
    productL = $("#productP").find("input")
    productL.each(function (index, item) {
        product += item.value + '。'
    })
    //获得员工信息
    workerId = $("input[name='员工身份证号']").val()
    businessTravelTime = $("input[name='商业旅行时间']").val()
    monthlyIncome = $("input[name='月收入']").val()
    companies = $("input[name='任职过多少公司']").val()
    onTheJob = $("input[name='在职时常']").val()

    
    data = {
        "product": product,
        "businessTravelTime": businessTravelTime,
        "workerId": workerId,
        "monthlyIncome": monthlyIncome,
        "companies": companies,
        "onTheJob": onTheJob
    }
    $.ajax({
        url:"http://localhost:8000/result",
        type:"POST",
        data:data,
        timeout:30000,
        dataType:"json",
        success:function(data){
            $("#loading").css("display", "none");
            $('#cover').css('display','none'); //显示遮罩层
            $('#header').css('opacity',"1"); 
            response = data
            window.location.href = "result.html?product=" + response.data.product + "&team=" + response.data.team;
        },
        error:function(){
            alert("请求发送错误，请重新尝试");
            unLoading()
        }
      });
};

function getQuery(){
    $("#navTabs").css("display","none")
    $("#contentMain1").css("display","block")
    $("#contentMain2").css("display","none")
    $("#getQuery").addClass("active")
    $("#getInput").removeClass("active")
}
function getInput(){
    $("#navTabs").css("display","block")
    $("#contentMain1").css("display","none")
    $("#contentMain2").css("display","block")
    $("#getInput").addClass("active")
    $("#getQuery").removeClass("active")
}

function queryByCompanyName(){
    Loading()                           //加载
    companyName = $("input[name='企业名称']").val()
    data = {
        "companyName": companyName
    }
    $.ajax({
        url:"http://localhost:8000/queryByCompanyName",
        type:"POST",
        data:data,
        timeout:30000,
        dataType:"json",
        success:function(data){
            if(data.data.product!="0"){
                response = data
                window.location.href = "result.html?product=" + response.data.product + "&team=" + response.data.team;
            }
            alert("该企业不存在于数据库中，请重新尝试");
            unLoading()
        },
        error:function(){
            alert("请求发送错误，请重新尝试");
            unLoading()
        }
      });
}