

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
$(function(){
	team =  $(".section_content").find("span")[0].innerHTML = parseInt(getUrlQueryString('team') * 100)
	if (team <= 50){team = team + 50}
	$(".section_content").find("span")[0].innerHTML = team
	$(".section_content").find("span")[1].innerHTML = getUrlQueryString('product') 
	//document.querySelector("body > div.section-wrap.put-section-1 > div.section.section-1 > div > div > div > div:nth-child(1) > div:nth-child(3) > strong > span").innerHTML = 89
	$(".section_content").find("span")[3].innerHTML= ((team+parseInt(getUrlQueryString('product'))+89)/3).toFixed(2)
	});