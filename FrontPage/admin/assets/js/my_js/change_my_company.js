var company_info
var companyId = getUrlQueryString('companyId')
args = {
    "companyId": companyId
}
post('getCompanyInfo', args,
    (data) => {
        data = JSON.parse(data)
        company_info = data.data
        insert_info(company_info)
    },
    (error) => alert("请求错误")
)
function insert_info(company_info) {
    for(key in company_info){
        value = company_info[key]
        template = "input[name='"+     key     +"']"
        $(template).val(value)
    }
}


function chang_my_company(){
    for(key in company_info){
        template = "input[name='"+     key     +"']"
        company_info[key] = $(template).val()
    }
    post('changeMyCompanyInfo', company_info,
    (data) => {
        alert("修改成功")
    },
    (error) => alert("请求错误")
)
}

function getUrlQueryString(names, urls) {
	urls = urls || window.location.href;
	urls && urls.indexOf("?") > -1 ? urls = urls
			.substring(urls.indexOf("?") + 1) : "";
	var reg = new RegExp("(^|&)" + names + "=([^&]*)(&|$)", "i");
	var r = urls ? urls.match(reg) : window.location.search.substr(1)
			.match(reg);
	if (r != null && r[2] != "")
		return unescape(r[2]);
	return null;
};
