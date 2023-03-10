function switch_to_tab(tab){
  var res=-1;
  var done=false;
  gradioApp().querySelector('#tabs').childNodes.forEach((x)=>{
    if(x.nodeName == "DIV"){
      if(x.id==tab){
        done = true
      }else if(!done){
        res += 1;
      }
    }
  })
  console.log(res)
  var btn;
  gradioApp().querySelector('#tabs').querySelectorAll('button').forEach((x)=>{
    if(res==0){
      btn = x;
      x.click();
      res -= 1;
    }else{
      res -= 1;
    }
  })
  console.log(btn)
  
  return args_to_array(arguments);
}


function switch_to_haku_img(){
  switch_to_tab('tab_haku_img')
  return switch_to_haku_blend()
}
function switch_to_haku_img_eff(){
  switch_to_tab('tab_haku_img')
  return switch_to_haku_eff()
}

function switch_to_inner_tab(tab){
  var res=-1;
  var done=false;
  gradioApp().querySelector('#haku_blend_tabs').childNodes.forEach((x)=>{
    if(x.nodeName == "DIV"){
      if(x.id==tab){
        done = true
      }else if(!done){
        res += 1;
      }
    }
  })
  console.log(res)
  gradioApp().querySelector('#haku_blend_tabs').querySelectorAll('button').forEach((x)=>{
    if(res==0){
      x.click();
      console.log(x)
    }
    res -= 1;
  })
  
  return args_to_array(arguments);
}
function switch_to_haku_blend(){
  return switch_to_inner_tab('haku_blend')
}
function switch_to_haku_eff(){
  return switch_to_inner_tab('haku_eff')
}


function switch_to_inpaint_upload(){
  switch_to_img2img_tab(4);
  return args_to_array(arguments);
}