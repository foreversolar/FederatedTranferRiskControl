window.Apex = {
  chart: {
    foreColor: '#333',
    toolbar: {
      show: false
    },
  },
  stroke: {
    width: 3
  },
  dataLabels: {
    enabled: false
  }
};


var companyId = getUrlQueryString('companyId')
//图表三渲染
var my_company;
var list_date;
args = {
    "companyId": companyId
}
post('getCompanyAllScore', args,
  (data) => {
    data = JSON.parse(data)
    my_company = data.data.data
    list_date = data.data.date
  },
  (error) => alert(error)
)

var optionsLine = {
  chart: {
    height: 360,
    type: 'line',
    zoom: {
      enabled: false
    },
    toolbar: {
      show: false
    },
    dropShadow: {
      enabled: true,
      top: 3,
      left: 2,
      blur: 4,
      opacity: .5,
    }
  },
  stroke: {
    curve: 'smooth',
    width: 2
  },
  //colors: ["#3F51B5", '#2196F3'],
  series: my_company,
  title: {
    text: 'Media',
    align: 'left',
    offsetY: 25,
    offsetX: 20
  },
  subtitle: {
    text: 'Statistics',
    offsetY: 55,
    offsetX: 20
  },
  markers: {
    size: 6,
    strokeWidth: 0,
    hover: {
      size: 9
    }
  },
  grid: {
    show: true,
  },
  labels: list_date,
  xaxis: {
    type: 'datetime',
    tooltip: {
      enabled: false
    }
  },
  legend: {
    position: 'top',
    horizontalAlign: 'right',
    offsetY: -20
  }
}

var chartLine = new ApexCharts(document.querySelector('#line-adwords'), optionsLine);
chartLine.render();









//图表一渲染
var product_scores;
var team_scores;
var company_scores;
var company_name;
args = {
  "companyId": companyId
}
post('getCompanyScore', args,
  (data) => {
    data = JSON.parse(data)
    product_scores = data.data.product_scores
    team_scores = data.data.product_scores
    company_scores  = data.data.company_scores
    company_name = data.data.company_name
  },
  (error) => alert(error)
)
var optionsBar = {
  chart: {
    height: 380,
    type: 'bar',
    stacked: true,
  },
  plotOptions: {
    bar: {
      columnWidth: '30%',
      horizontal: false,
    },
  },
  series: [{
    name: '公司得分',
    data: company_scores
  }, {
    name: '团队得分',
    data: team_scores
  }, {
    name: '产品得分',
    data: product_scores
  }],
  xaxis: {
    categories: company_name,
  },
  fill: {
    opacity: 1
  },

}
var chartBar = new ApexCharts(
  document.querySelector("#barchart"),
  optionsBar
);
chartBar.render();




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