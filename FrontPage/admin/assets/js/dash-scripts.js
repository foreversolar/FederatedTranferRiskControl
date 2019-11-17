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

var userId = localStorage.getItem("userId")
//图表三渲染
var my_company;
args = {
  "userId": userId
}
post('getMyCompanyScore', args,
  (data) => {
    data = JSON.parse(data)
    my_company = data.data.data
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
  labels: ['01/15/2002', '02/16/2002', '03/17/2002', '04/18/2002', '05/19/2002', '06/20/2002'],
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
args = {
  "userId": userId
}
post('getAllMyCompany', args,
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




//图表四渲染
var company_list
args = {
  "userId": userId
}
post('getMyCompanyStatus', args,
  (data) => {
    data = JSON.parse(data)
    company_list = data.data.data
    company_list.forEach(item => {
      company_template(item[0],item[1],item[2],item[3])
    });
  },
  (error) => alert(error)
)
function company_template(name,status,level,score){
    t = "<tr><td><div class='round img2'><span class='colored-block gradient-blue'></span></div><div class='designer-info'><h6>"
    t += name
    t += "</h6></div></td><td><span class='badge badge-md w-70 round-success'>"
    t += status
    t += "</span></td><td class='doc-rating'>"
    for(i=0;i<level;i++){
      t += "<i class='fa fa-star'></i>"
    }
    t += "<span>" + level + "</span>"
    t += "</td><td class='text-center'>"
    t += score
    t += "</td></tr>"
    
    if(status=="良好")t=t.replace(/round-success/g,"gradient-orange")
    else if(status=="一般")t=t.replace(/round-success/g,"round-danger")
    $("#myCompany").append(t);
}


//图表四渲染
var follow_company_list
args = {
  "userId": userId
}
post('getMyFollowCompanyStatus', args,
  (data) => {
    data = JSON.parse(data)
    follow_company_list = data.data.data
    follow_company_list.forEach(item => {
      follow_company_template(item[0],item[1],item[2],item[3])
    });
  },
  (error) => alert(error)
)
function follow_company_template(name,status,level,score){
    t = "<tr><td><div class='round img2'><span class='colored-block gradient-blue'></span></div><div class='designer-info'><h6>"
    t += name
    t += "</h6></div></td><td><span class='badge badge-md w-70 round-success'>"
    t += status
    t += "</span></td><td class='doc-rating'>"
    for(i=0;i<level;i++){
      t += "<i class='fa fa-star'></i>"
    }
    t += "<span>" + level + "</span>"
    t += "</td><td class='text-center'>"
    t += score
    t += "</td></tr>"
    if(status=="良好")t=t.replace(/round-success/g,"gradient-orange")
    else if(status=="一般")t=t.replace(/round-success/g,"round-danger")
    $("#followCompany").append(t);
}
