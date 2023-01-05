function change_img_height(id, height){
  query = id+', '+id+' .h-60, '+id+' .h-60 div, '+id+' .h-60 div img'
  console.log(query)
  gradioApp().querySelectorAll(query).forEach((x)=>{
    x.style.height = Math.floor(height)+'px'
    x.style['max-height'] = Math.floor(height)+'px'
  })
  
  query = id+' canvas'
  gradioApp().querySelectorAll(query).forEach((x)=>{
    x.style['max-height'] = height+'px'
    x.style['max-width'] = '100%'
    x.style.height = 'auto'
    x.style.width = 'auto'
  })
}

function get_change_height(id){
  return (height)=>{change_img_height(id, height)}
}