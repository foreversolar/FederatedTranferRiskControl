var all_company_list
var userId = localStorage.getItem("userId")
args = {
    "userId": userId
}
post('getMyFollowCompanyStatus', args,
    (data) => {
        data = JSON.parse(data)
        all_company_list = data.data.data

        all_company_list.forEach(item => {
            all_company_template(item[0], item[1], item[2], item[3])
        });
    },
    (error) => alert(error)
)
function all_company_template(name, status, level, score) {
    t = "<tr><td><div class='round img2'><span class='colored-block gradient-blue'></span></div><div class='designer-info'><h6>"
    t += name
    t += "</h6></div></td><td><span class='badge badge-md w-70 round-success'>"
    t += status
    t += "</span></td><td class='doc-rating'>"
    for (i = 0; i < level; i++) {
        t += "<i class='fa fa-star'></i>"
    }
    t += "<span>" + level + "</span>"
    t += "</td><td class='text-center'>"
    t += score
    t += "</td>"
    t += "<td><button class='btn btn-purple' onclick='cancerFollow(this)'><i class='fa fa-book'></i></button></td>"
    t += "<td><button class='btn btn-purple' onclick='result(this)'><i class='fa fa-dashboard'></i></button></td>"
    t += "</tr>"
    if (status == "良好") t = t.replace(/round-success/g, "gradient-orange")
    else if (status == "一般") t = t.replace(/round-success/g, "round-danger")
    $("#followCompany").append(t);
}


function cancerFollow(e) {
    td = $(e).parents('tr').children('td')
    companyName = $(td).eq(0).text()                  //获得公司名称
    console.log(companyName)
    args = {
        "userId": 1,
        "companyName": companyName
    }
    post('cancerFollowCompany', args,
        (data) => {
            alert("取消成功")
        },
        (error) => alert(error)
    )
}



function result(e){
    td = $(e).parents('tr').children('td')
    companyName = $(td).eq(0).text()                  //获得公司名称
    console.log(companyName)
    var companyId
    post('getCompanyId', {"companyName":companyName},                  //获得公司id
    (data) => {
        data = JSON.parse(data)
        companyId = data.data.companyId
    },
    (error) => alert("请求发生错误")
)

    window.location.href = "result.html?companyId=" + companyId;
}