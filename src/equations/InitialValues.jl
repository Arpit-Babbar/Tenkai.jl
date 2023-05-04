module InitialValues

function flat_hat2d(x)
   if 0.25 < x[1] < 0.75 && -0.25 < x[2] < 0.25
      return 1.0
   else
      return 0.0
   end
end

function smooth_hump2d(x)
   # r0 = 0.15
   # d = sqrt( (x[1]-0.25)^2+(x[2]-0.5)^2 )
   # q = min(d, r0)/r0
   # return 0.25*(1.0+cospi(q))
   x, y = x
   r0 = 0.15
   d = sqrt( (x-0.25)^2+(y-0.5)^2 )
   q = min(d, r0)/r0
   return 0.25*(1.0+cospi(q))
end

function cone2d(x)
   x, y = x
   # r0 = 0.15
   # d  = sqrt( (x[1]-0.5)^2+(x[2]-0.25)^2 )
   # if d < r0
   #    return 1.0-d/r0
   # else
   #    return 0.0
   # end
   r0 = 0.15
   d  = sqrt( (x-0.5)^2+(y-0.25)^2 )
   if d < r0
      return 1.0-d/r0
   else
      return 0.0
   end
end

function slotted_disc2d(x)
   # r0 = 0.15
   # d = sqrt( (x[1]-0.5)^2+(x[2]-0.75)^2 )
   # if d < r0
   #    if (x[1]> 0.5-r0*0.25 && x[1] < 0.5+r0*0.25) && x[2] < 0.75+r0*0.7
   #       return 0.0
   #    else
   #       return  1.0
   #    end
   # else
   #    return 0.0
   # end
   x, y = x
   r0 = 0.15
   d = sqrt( (x-0.5)^2+(y-0.75)^2 )
   if d < r0
      if (x> 0.5-r0*0.25 && x < 0.5+r0*0.25) && y < 0.75+r0*0.7
         return 0.0
      else
         return  1.0
      end
   else
      return 0.0
   end
end

function composite2d(x)
   return smooth_hump2d(x)+cone2d(x)+slotted_disc2d(x)
end

export flat_hat1d
export flat_hat2d
export composite2d


export flat_hat1d
export flat_hat2d
export composite2d
end
